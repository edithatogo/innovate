#include <string.h>
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#ifdef _WIN32
#include <windows.h>
static void *open_native_library(SEXP path) {
    const char *utf8 = Rf_translateCharUTF8(path);
    int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, NULL, 0);
    if (!length) Rf_error("invalid UTF-8 library path");
    wchar_t *wide = (wchar_t *) R_alloc((size_t) length, sizeof(wchar_t));
    if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, wide, length))
        Rf_error("unable to convert native library path");
    return LoadLibraryW(wide);
}
#define OPEN(path) open_native_library(path)
#define SYMBOL(handle, name) GetProcAddress(handle, name)
#define CLOSE(handle) FreeLibrary(handle)
#else
#include <dlfcn.h>
#define OPEN(path) dlopen(Rf_translateCharUTF8(path), RTLD_NOW | RTLD_LOCAL)
#define SYMBOL(handle, name) dlsym(handle, name)
#define CLOSE(handle) dlclose(handle)
#endif

static SEXP call_kernel(SEXP library, SEXP request, SEXP data) {
    if (TYPEOF(library) != STRSXP || XLENGTH(library) != 1 || STRING_ELT(library, 0) == NA_STRING ||
        TYPEOF(request) != STRSXP || XLENGTH(request) != 1 || STRING_ELT(request, 0) == NA_STRING)
        Rf_error("library and request must be scalar strings");
    void *handle = OPEN(STRING_ELT(library, 0));
    if (!handle) Rf_error("unable to load the Innovate native library");
    void (*release)(char *) = (void (*)(char *)) SYMBOL(handle, "innovate_free");
    if (!release) { CLOSE(handle); Rf_error("incompatible Innovate native ABI"); }
    const char *text = Rf_translateCharUTF8(STRING_ELT(request, 0));
    char *response;
    if (data == R_NilValue) {
        char *(*call)(const char *) = (char *(*)(const char *)) SYMBOL(handle, "innovate_call");
        if (!call) { CLOSE(handle); Rf_error("incompatible Innovate native ABI"); }
        response = call(text);
    } else {
        char *(*call)(const unsigned char *, size_t, const char *) =
            (char *(*)(const unsigned char *, size_t, const char *)) SYMBOL(handle, "innovate_scenario_grid_ipc");
        if (!call) { CLOSE(handle); Rf_error("native library lacks scenario Arrow IPC support"); }
        response = call(RAW(data), (size_t) XLENGTH(data), text);
    }
    if (!response) { CLOSE(handle); Rf_error("native response allocation failed"); }
    SEXP result = PROTECT(Rf_ScalarString(Rf_mkCharCE(response, CE_UTF8)));
    release(response);
    CLOSE(handle);
    UNPROTECT(1);
    return result;
}

SEXP innovate_call_R(SEXP library, SEXP request) {
    return call_kernel(library, request, R_NilValue);
}

SEXP innovate_scenario_grid_ipc_R(SEXP library, SEXP payload, SEXP data) {
    if (TYPEOF(data) != RAWSXP) Rf_error("data must be a raw Arrow IPC vector");
    return call_kernel(library, payload, data);
}

static SEXP scenario_output(SEXP library, SEXP response, const char *symbol) {
    int binary = strcmp(symbol, "innovate_export_ipc_file") == 0;
    if (TYPEOF(library) != STRSXP || XLENGTH(library) != 1 || STRING_ELT(library, 0) == NA_STRING)
        Rf_error("library must be a scalar string");
    if (binary ? TYPEOF(response) != RAWSXP :
        (TYPEOF(response) != STRSXP || XLENGTH(response) != 1 || STRING_ELT(response, 0) == NA_STRING))
        Rf_error("expected raw IPC bytes or a scalar scenario response string");
    void *handle = OPEN(STRING_ELT(library, 0));
    if (!handle) Rf_error("unable to load the Innovate native library");
    void *convert = (void *) SYMBOL(handle, symbol);
    void (*free_bytes)(unsigned char *, size_t) =
        (void (*)(unsigned char *, size_t)) SYMBOL(handle, "innovate_free_bytes");
    void (*free_text)(char *) = (void (*)(char *)) SYMBOL(handle, "innovate_free");
    if (!convert || !free_bytes || !free_text) {
        CLOSE(handle); Rf_error("native library lacks Arrow export support");
    }
    unsigned char *data = NULL;
    size_t len = 0;
    char *failure = binary
        ? ((char *(*)(const unsigned char *, size_t, unsigned char **, size_t *)) convert)
            (RAW(response), (size_t) XLENGTH(response), &data, &len)
        : ((char *(*)(const char *, unsigned char **, size_t *)) convert)
            (Rf_translateCharUTF8(STRING_ELT(response, 0)), &data, &len);
    if (failure) {
        SEXP result = PROTECT(Rf_ScalarString(Rf_mkCharCE(failure, CE_UTF8)));
        free_text(failure);
        CLOSE(handle);
        UNPROTECT(1);
        return result;
    }
    if (!data || len > R_XLEN_T_MAX) {
        free_bytes(data, len);
        CLOSE(handle);
        Rf_error("invalid native Arrow output size");
    }
    SEXP result = PROTECT(Rf_allocVector(RAWSXP, (R_xlen_t) len));
    memcpy(RAW(result), data, len);
    free_bytes(data, len);
    CLOSE(handle);
    UNPROTECT(1);
    return result;
}

SEXP innovate_scenario_executions_ipc_R(SEXP library, SEXP response) {
    return scenario_output(library, response, "innovate_scenario_executions_ipc");
}

SEXP innovate_scenario_comparisons_ipc_R(SEXP library, SEXP response) {
    return scenario_output(library, response, "innovate_scenario_comparisons_ipc");
}

SEXP innovate_paired_outcomes_ipc_R(SEXP library, SEXP payload) {
    return scenario_output(library, payload, "innovate_paired_outcomes_ipc");
}


SEXP innovate_export_ipc_file_R(SEXP library, SEXP input) {
    return scenario_output(library, input, "innovate_export_ipc_file");
}

static const R_CallMethodDef methods[] = {
    {"innovate_export_ipc_file_R", (DL_FUNC) &innovate_export_ipc_file_R, 2},
    {"innovate_call_R", (DL_FUNC) &innovate_call_R, 2},
    {"innovate_scenario_grid_ipc_R", (DL_FUNC) &innovate_scenario_grid_ipc_R, 3},
    {"innovate_scenario_executions_ipc_R", (DL_FUNC) &innovate_scenario_executions_ipc_R, 2},
    {"innovate_paired_outcomes_ipc_R", (DL_FUNC) &innovate_paired_outcomes_ipc_R, 2},
    {"innovate_scenario_comparisons_ipc_R", (DL_FUNC) &innovate_scenario_comparisons_ipc_R, 2},
    {NULL, NULL, 0}
};

void R_init_innovate_R(DllInfo *dll) {
    R_registerRoutines(dll, NULL, methods, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
