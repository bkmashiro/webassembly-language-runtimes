/*
 * py_reactor.c – CPython WASM reactor exports for shimmy-wasm
 *
 * This file is compiled against libpython3.14.a (the static library built from
 * CPython 3.14 cross-compiled for wasm32-wasip1) and linked with
 * -mexec-model=reactor, producing a WASM module that exposes a callable
 * function interface instead of the traditional command-mode _start.
 *
 * Exported functions
 * ------------------
 *
 *   py_init()               Called once by the host after _initialize.
 *                           Starts CPython and defines _handle_request in
 *                           the __main__ namespace.
 *
 *   py_exec(ptr, len)       Execute one request.  ptr/len describe a JSON
 *                           string in linear memory.  Result is written to
 *                           _resp_buf; length to _resp_len.
 *
 *   alloc(size) → ptr       Allocate scratch space in WASM linear memory.
 *                           Used by the host to pass request JSON in.
 *
 *   dealloc(ptr)            Free scratch space allocated with alloc().
 *
 *   resp_buf()  → ptr       Address of the response byte buffer (4 MiB).
 *
 *   resp_len()  → ptr       Address of the int32_t response length field.
 *
 * Host protocol (per request)
 * ---------------------------
 *
 *   // ① restore memory snapshot (safe: no WASM goroutine is running)
 *   restoreSnapshot(mod)
 *
 *   // ② write request JSON into WASM linear memory
 *   ptr, _ := mod.ExportedFunction("alloc").Call(ctx, uint64(len(reqJSON)))
 *   mod.Memory().Write(uint32(ptr[0]), reqJSON)
 *
 *   // ③ execute
 *   mod.ExportedFunction("py_exec").Call(ctx, ptr[0], uint64(len(reqJSON)))
 *
 *   // ④ read result
 *   bufPtrV, _ := mod.ExportedFunction("resp_buf").Call(ctx)
 *   lenPtrV, _ := mod.ExportedFunction("resp_len").Call(ctx)
 *   respLen, _ := mod.Memory().ReadUint32Le(uint32(lenPtrV[0]))
 *   result, _  := mod.Memory().Read(uint32(bufPtrV[0]), respLen)
 *
 *   // ⑤ free scratch
 *   mod.ExportedFunction("dealloc").Call(ctx, ptr[0])
 *
 * Why reactor mode enables safe snapshot / restore
 * ------------------------------------------------
 *
 * In command mode (current python.wasm), a long-running server loop keeps a
 * Go goroutine permanently blocked inside WASM execution.  Restoring linear
 * memory while that goroutine holds live WASM stack frames in its Go call
 * stack causes "out of bounds memory access" crashes.
 *
 * In reactor mode, py_exec() RETURNS after each request.  Between calls the
 * WASM goroutine is fully unwound back to Go.  Linear memory is the only
 * WASM state; __stack_pointer is at its post-_initialize baseline.  Both
 * can be safely snapshotted and restored without any goroutine conflict.
 *
 * NumPy static built-in support
 * ------------------------------
 *
 * When compiled with -DHAVE_NUMPY, numpy's C extension modules are registered
 * as built-in modules via PyImport_AppendInittab() before Py_Initialize().
 * This makes numpy importable without dynamic linking (.so files) — the
 * extension symbols are linked directly into this WASM binary alongside
 * libpython3.14.a and the numpy static archives (.a files).
 */

#include <Python.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/* ── NumPy built-in module registration ──────────────────────────────────── */

#ifdef HAVE_NUMPY
/*
 * Only numpy.core, numpy.fft, and numpy.linalg C extensions are registered
 * as Python built-in modules.  numpy.random C extensions are intentionally
 * NOT registered here.
 *
 * Rationale: The Cython-compiled numpy.random extensions (bit_generator,
 * _mt19937, mtrand, etc.) call Py_FatalError() → abort() → WASM unreachable
 * during their PyInit_* functions when run inside the WASI sandbox.  The
 * exact trigger is Cython's vtable initialisation dereferencing function
 * pointers that are not valid in the WASI execution environment.
 *
 * Instead, all numpy.random.* modules are pre-populated as pure-Python stubs
 * in initSysPathScript (python_reactor.go) before numpy is imported.  Because
 * Python's import system checks sys.modules before calling any finder or init
 * function, the broken C initialisers are never reached.
 *
 * numpy.core, numpy.linalg, and numpy.fft work correctly; only the
 * new-style Generator API (np.random.default_rng, np.random.Generator) is
 * unavailable.  The legacy np.random functions (seed, rand, randn, …) return
 * a NotImplementedError with a clear message.
 */
extern PyObject *PyInit__multiarray_umath(void);
extern PyObject *PyInit__pocketfft_internal(void);
extern PyObject *PyInit__umath_linalg(void);
extern PyObject *PyInit_lapack_lite(void);

static void register_numpy_builtins(void) {
    PyImport_AppendInittab("numpy.core._multiarray_umath",  PyInit__multiarray_umath);
    PyImport_AppendInittab("numpy.fft._pocketfft_internal", PyInit__pocketfft_internal);
    PyImport_AppendInittab("numpy.linalg._umath_linalg",    PyInit__umath_linalg);
    PyImport_AppendInittab("numpy.linalg.lapack_lite",      PyInit_lapack_lite);
    /* numpy.random.* — stubbed in Python; see python_reactor.go initSysPathScript */
}
#endif /* HAVE_NUMPY */

/* 4 MiB response buffer – more than enough for any eval result. */
#define RESP_BUF_SIZE (4 * 1024 * 1024)

static char    _resp_buf[RESP_BUF_SIZE];
static int32_t _resp_len = 0;

/* ── Python handler source ───────────────────────────────────────────────── */

/*
 * Defined once during py_init().  Each call to _handle_request():
 *
 *   • parses the request JSON
 *   • execs the user script in a fresh namespace (exec with ns={})
 *   • calls evaluation_function / preview_function
 *   • evicts any modules the user script imported from sys.modules
 *     (prevents cross-request module-state leaks)
 *   • returns the result as a JSON string
 */
static const char *HANDLER_SRC =
    "import sys as _sys, json as _json, importlib.util as _ilu, importlib.machinery as _ilm, _imp as _imp_mod\n"
    "\n"
    "# ── wasi-vfs finder ──────────────────────────────────────────────────────\n"
    "# Two problems this finder solves:\n"
    "#\n"
    "# 1. wasi-vfs intercepts path_open (file reads) but NOT fd_readdir (scandir).\n"
    "#    Python's FileFinder calls os.scandir() on sys.path entries → fails\n"
    "#    silently, making all VFS-packed packages (numpy pure Python) invisible.\n"
    "#    Fix: probe file existence with open() instead of scandir().\n"
    "#\n"
    "# 2. BuiltinImporter.find_spec() returns None when `path is not None`\n"
    "#    (i.e. when importing a submodule).  So C extensions registered with\n"
    "#    PyImport_AppendInittab as dotted names like\n"
    "#    'numpy.core._multiarray_umath' are skipped by BuiltinImporter when\n"
    "#    imported as numpy submodules, and PathFinder tries to dlopen() them —\n"
    "#    which WASI does not support.  Fix: check _imp.is_builtin() ourselves\n"
    "#    and return a BuiltinImporter spec so Python loads them from the\n"
    "#    already-linked symbol table.\n"
    "_SITE = '/usr/lib/python3.14/site-packages'\n"
    "\n"
    "class _WasivfsFinder:\n"
    "    @staticmethod\n"
    "    def _exists(p):\n"
    "        try:\n"
    "            open(p, 'rb').close(); return True\n"
    "        except OSError:\n"
    "            return False\n"
    "    def find_spec(self, fullname, path, target=None):\n"
    "        # Built-in C extension registered via PyImport_AppendInittab?\n"
    "        # BuiltinImporter skips submodule imports (path is not None), so we\n"
    "        # must handle them here to prevent PathFinder from attempting dlopen.\n"
    "        if _imp_mod.is_builtin(fullname):\n"
    "            return _ilu.spec_from_loader(fullname, _ilm.BuiltinImporter)\n"
    "        # Pure Python package / module packed into wasi-vfs.\n"
    "        parts = fullname.split('.')\n"
    "        base  = _SITE + '/' + '/'.join(parts)\n"
    "        init  = base + '/__init__.py'\n"
    "        if self._exists(init):\n"
    "            loader = _ilm.SourceFileLoader(fullname, init)\n"
    "            return _ilu.spec_from_file_location(\n"
    "                fullname, init, loader=loader,\n"
    "                submodule_search_locations=[base])\n"
    "        src = base + '.py'\n"
    "        if self._exists(src):\n"
    "            loader = _ilm.SourceFileLoader(fullname, src)\n"
    "            return _ilu.spec_from_file_location(fullname, src, loader=loader)\n"
    "        return None\n"
    "\n"
    "if _SITE not in _sys.path:\n"
    "    _sys.path.insert(0, _SITE)\n"
    "# Remove PathFinder from its current position, append our finder, then\n"
    "# re-append PathFinder.  Guarantees order:\n"
    "#   [BuiltinImporter, FrozenImporter, _WasivfsFinder, PathFinder]\n"
    "# regardless of what has already been done to meta_path.\n"
    "_sys.meta_path = [_f for _f in _sys.meta_path if _f is not _ilm.PathFinder]\n"
    "_sys.meta_path.append(_WasivfsFinder())\n"
    "_sys.meta_path.append(_ilm.PathFinder)\n"
    "\n"
    "# ── Stub numpy test-only C extensions ────────────────────────────────────\n"
    "# numpy/core/_add_newdocs.py imports test-only extensions\n"
    "# (e.g. numpy.core._multiarray_tests) to attach docstrings via add_newdoc().\n"
    "# These are not compiled into the WASI binary.  Stub them BEFORE capturing\n"
    "# _base_modules so they are treated as 'baseline' and never evicted.\n"
    "import types as _types\n"
    "_NUMPY_STUBS = [\n"
    "    'numpy.core._multiarray_tests',\n"
    "    'numpy.core._umath_tests',\n"
    "    'numpy.core._rational_tests',\n"
    "    'numpy.core._struct_ufunc_tests',\n"
    "    'numpy.core._operand_flag_tests',\n"
    "]\n"
    "for _n in _NUMPY_STUBS:\n"
    "    if _n not in _sys.modules:\n"
    "        _m = _types.ModuleType(_n)\n"
    "        # _add_newdocs.py calls getattr(module, func_name) to attach docstrings.\n"
    "        # The module has no real functions so __getattr__ returns a dummy callable.\n"
    "        _m.__getattr__ = lambda _attr: (lambda *_a, **_kw: None)\n"
    "        _sys.modules[_n] = _m\n"
    "\n"
    "# Capture sys.modules state after stdlib import so we can evict\n"
    "# any modules the user script imports between requests.\n"
    "_base_modules = frozenset(_sys.modules.keys())\n"
    "\n"
    "def _handle_request(req_json):\n"
    "    req        = _json.loads(req_json)\n"
    "    script_src = req.get('script', '')\n"
    "    input_data = req.get('input', {})\n"
    "    method     = req.get('method', 'eval')\n"
    "    try:\n"
    "        ns = {}\n"
    "        exec(compile(script_src, '<eval>', 'exec'), ns)\n"
    "        if method == 'preview':\n"
    "            fn = ns.get('preview_function') or ns.get('evaluation_function')\n"
    "        else:\n"
    "            fn = ns.get('evaluation_function')\n"
    "        if fn is None:\n"
    "            result = {'error': 'no evaluation_function defined in script'}\n"
    "        else:\n"
    "            result = fn(\n"
    "                input_data.get('response'),\n"
    "                input_data.get('answer'),\n"
    "                input_data.get('params', {})\n"
    "            )\n"
    "    except Exception as _e:\n"
    "        result = {'error': str(_e)}\n"
    "    finally:\n"
    "        # Evict user-imported modules so their state does not bleed\n"
    "        # into the next request (even if snapshot/restore is not used).\n"
    "        for _k in list(_sys.modules.keys()):\n"
    "            if _k not in _base_modules:\n"
    "                del _sys.modules[_k]\n"
    "    return _json.dumps(result)\n";

/* ── Exported: initialisation ────────────────────────────────────────────── */

__attribute__((export_name("py_init")))
void py_init(void) {
#ifdef HAVE_NUMPY
    /* Register numpy C extension modules as built-ins BEFORE Py_Initialize().
     * PyImport_AppendInittab() must be called before the interpreter starts. */
    register_numpy_builtins();
#endif

    Py_Initialize();
    if (PyRun_SimpleString(HANDLER_SRC) != 0) {
        /*
         * If the handler source fails to compile/run something is very wrong.
         * Abort loudly so the host sees a non-zero exit code rather than a
         * silently broken module.
         */
        Py_Finalize();
        __builtin_trap();
    }
}

/* ── Exported: memory helpers ────────────────────────────────────────────── */

__attribute__((export_name("alloc")))
char *alloc(int32_t size) { return malloc((size_t)size); }

__attribute__((export_name("dealloc")))
void dealloc(char *ptr) { free(ptr); }

__attribute__((export_name("resp_buf")))
char *resp_buf(void) { return _resp_buf; }

__attribute__((export_name("resp_len")))
int32_t *resp_len(void) { return &_resp_len; }

/* ── Exported: request handler ───────────────────────────────────────────── */

static void write_error(const char *msg) {
    int n = snprintf(_resp_buf, RESP_BUF_SIZE, "{\"error\":\"%s\"}", msg);
    _resp_len = (n > 0 && n < RESP_BUF_SIZE) ? n : 0;
}

/*
 * py_exec(req, req_len) – execute one eval/preview request.
 *
 * req     : pointer to request JSON bytes in WASM linear memory
 * req_len : byte length of req
 *
 * On return, the JSON result is in _resp_buf[0.._resp_len).
 * The host reads it via resp_buf() / resp_len().
 */
__attribute__((export_name("py_exec")))
void py_exec(char *req, int32_t req_len) {
    _resp_len = 0;

    PyObject *main_mod = PyImport_AddModule("__main__");
    if (!main_mod) {
        write_error("py_exec: cannot access __main__");
        return;
    }

    PyObject *handler = PyObject_GetAttrString(main_mod, "_handle_request");
    if (!handler) {
        PyErr_Clear();
        write_error("py_exec: _handle_request not found — was py_init called?");
        return;
    }

    PyObject *py_req = PyUnicode_FromStringAndSize(req, (Py_ssize_t)req_len);
    if (!py_req) {
        PyErr_Clear();
        Py_DECREF(handler);
        write_error("py_exec: failed to decode request as UTF-8");
        return;
    }

    PyObject *result = PyObject_CallOneArg(handler, py_req);
    Py_DECREF(py_req);
    Py_DECREF(handler);

    if (!result) {
        PyErr_Clear();
        write_error("py_exec: _handle_request raised an unhandled exception");
        return;
    }

    Py_ssize_t out_len;
    const char *out_str = PyUnicode_AsUTF8AndSize(result, &out_len);
    if (out_str && out_len < (Py_ssize_t)RESP_BUF_SIZE) {
        memcpy(_resp_buf, out_str, (size_t)out_len);
        _resp_len = (int32_t)out_len;
    } else {
        write_error("py_exec: result too large or encoding failed");
    }
    Py_DECREF(result);
}
