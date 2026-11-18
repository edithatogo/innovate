//! Narrow C ABI for Julia and R, with explicit borrowed-input and owned-output rules.
//!
//! JSON calls return Rust-owned, NUL-terminated UTF-8. Copy or inspect the text and
//! release it exactly once with [`innovate_free`]. Arrow exporters return a null
//! error pointer on success and write a Rust-owned byte pointer and byte length;
//! release that original pair with [`innovate_free_bytes`]. These buffers must not
//! be released by a foreign allocator. All input buffers are borrowed only for
//! the duration of the synchronous call.
//!
//! Error envelopes describe request or execution failures. They do not recover
//! from violations of an unsafe function's pointer preconditions. The ABI does
//! not infer scientific units or convert observation quantities.
use std::ffi::{CStr, CString, c_char};
use std::ptr;

/// Execute a schema-2.0 UTF-8 JSON request and return its response text.
///
/// # Returns
/// A Rust-owned, NUL-terminated JSON string, or null if the response cannot be
/// represented as a C string. Inspect the outer `error` member before consuming
/// `result`, and release every non-null response once with [`innovate_free`].
///
/// # Errors
/// Null input, invalid UTF-8 and rejected requests produce a native error
/// envelope. A caught Rust panic produces an `internal_error` envelope. Errors
/// retained inside batch result rows are distinct from the outer request error.
///
/// # Safety
/// A non-null `request` must point to a readable NUL-terminated string that
/// remains allocated and unchanged for this call. Null is accepted as an invalid
/// request; any other invalid pointer violates the contract.
///
/// # Examples
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use innovate_ffi::{innovate_call, innovate_free};
/// use std::ffi::{CStr, CString};
///
/// let request = CString::new(
///     r#"{"schema_version":"2.0","operation":"discover_models","payload":{}}"#,
/// )?;
/// // SAFETY: request is live and NUL-terminated for the synchronous call.
/// let response = unsafe { innovate_call(request.as_ptr()) };
/// assert!(!response.is_null());
/// // SAFETY: read the live response before returning its allocation to Rust.
/// let decoded = unsafe {
///     serde_json::from_slice::<serde_json::Value>(CStr::from_ptr(response).to_bytes())
/// };
/// // SAFETY: this is the original response pointer, released exactly once.
/// unsafe { innovate_free(response) };
/// let envelope = decoded?;
/// assert_eq!(envelope["schema_version"], "2.0");
/// assert!(envelope["error"].is_null());
/// assert!(envelope["result"]["models"].is_array());
/// # Ok(())
/// # }
/// ```
#[unsafe(no_mangle)]
pub unsafe extern "C" fn innovate_call(request: *const c_char) -> *mut c_char {
    let response = std::panic::catch_unwind(|| {
        if request.is_null() {
            return innovate_core::execute_json("");
        }
        // SAFETY: the caller's documented borrowed-string contract.
        match unsafe { CStr::from_ptr(request) }.to_str() {
            Ok(input) => innovate_core::execute_json(input),
            Err(_) => innovate_core::execute_json(""),
        }
    });
    owned_response(response)
}

/// Execute a scenario grid on a borrowed Arrow IPC file and return JSON.
///
/// `len` is the input buffer length in bytes. The payload is the scenario-grid
/// JSON object, not a full schema-2.0 request. The returned text follows the
/// allocation contract of [`innovate_call`] and must be freed with [`innovate_free`].
///
/// # Errors
/// Null inputs, a zero or oversized byte length, malformed JSON and rejected IPC
/// or grid data produce an `invalid_request` envelope. A caught Rust panic
/// produces `internal_error`. Inspect each returned scenario row as well as the
/// outer error before treating an execution as successful.
///
/// # Safety
/// Non-null inputs must remain allocated and unchanged for the call. `data` must
/// address `len` readable bytes in one allocation, and `payload` must address a
/// readable NUL-terminated string. No Rust reference may permit mutation of the
/// borrowed bytes while they are read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn innovate_scenario_grid_ipc(
    data: *const u8,
    len: usize,
    payload: *const c_char,
) -> *mut c_char {
    let invalid = |message: &str| {
        serde_json::json!({"error":{"code":"invalid_request","message":message},"result":null})
            .to_string()
    };
    owned_response(std::panic::catch_unwind(|| {
        if data.is_null() || len == 0 || len > isize::MAX as usize || payload.is_null() {
            return invalid("scenario grid requires Arrow IPC bytes and a JSON payload");
        }
        // SAFETY: caller supplies readable buffers for the duration of this call.
        let (bytes, text) = unsafe {
            (
                std::slice::from_raw_parts(data, len),
                CStr::from_ptr(payload),
            )
        };
        let Ok(payload) = serde_json::from_slice(text.to_bytes()) else {
            return invalid("invalid scenario grid JSON payload");
        };
        innovate_tabular::scenario_grid_ipc(bytes, payload)
            .unwrap_or_else(|error| invalid(&error.to_string()))
    }))
}

fn owned_response(response: std::thread::Result<String>) -> *mut c_char {
    let text = response.unwrap_or_else(|_| {
        r#"{"error":{"code":"internal_error","message":"native execution panicked"},"result":null}"#.into()
    });
    CString::new(text).map_or(ptr::null_mut(), CString::into_raw)
}

/// Release JSON text returned by this library. A null pointer is a no-op.
///
/// # Safety
/// `response` must be null or the original, still-live pointer returned by a
/// JSON call or an Arrow export's error return. Its string length and NUL
/// terminator must not have changed. Do not use the pointer after this call,
/// free it twice, or supply memory owned by another allocator.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn innovate_free(response: *mut c_char) {
    if !response.is_null() {
        // SAFETY: ownership is returned exactly once under the caller contract.
        drop(unsafe { CString::from_raw(response) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bass_dispatch_and_unknown_engine_cross_the_c_abi() {
        for engine in ["kairos_des", "unknown"] {
            let request = CString::new(serde_json::json!({
                "schema_version":"2.0","operation":"simulate_model","model_key":"bass",
                "payload":{"inputs":{"time":[0,1]},"parameters":{"p":1e12,"q":0,"m":1},
                    "simulation_options":{"engine":engine,"population":1,"seed":42,"ticks_per_unit":1}}
            }).to_string()).unwrap();
            // SAFETY: the input is live and NUL-terminated throughout the call.
            let out = unsafe { innovate_call(request.as_ptr()) };
            assert!(!out.is_null());
            // SAFETY: parse the live result before returning ownership to Rust.
            let parsed = unsafe {
                serde_json::from_slice::<serde_json::Value>(CStr::from_ptr(out).to_bytes())
            };
            unsafe { innovate_free(out) };
            let response = parsed.unwrap();
            assert_eq!(response["schema_version"], "2.0");
            if engine == "kairos_des" {
                assert!(response["error"].is_null());
                assert_eq!(response["result"]["counts"], serde_json::json!([0, 1]));
                assert_eq!(
                    response["result"]["metadata"]["kairos_dispatched_events"],
                    1
                );
            } else {
                assert!(response["result"].is_null());
                assert!(response["error"]["code"].is_string());
            }
        }
    }

    #[test]
    fn invalid_null_utf8_and_valid_requests_are_owned_and_released() {
        let valid =
            CString::new(r#"{"schema_version":"2.0","operation":"discover_models"}"#).unwrap();
        let malformed = CString::new("not json").unwrap();
        let invalid_utf8 = [255u8, 0];
        for input in [
            ptr::null(),
            valid.as_ptr(),
            malformed.as_ptr(),
            invalid_utf8.as_ptr().cast(),
        ] {
            // SAFETY: all non-null input allocations remain live and NUL-terminated.
            let out = unsafe { innovate_call(input) };
            assert!(!out.is_null());
            // SAFETY: returned allocation remains live until the explicit release.
            let text = unsafe { CStr::from_ptr(out) }.to_str().unwrap();
            assert!(text.contains("schema_version"));
            unsafe { innovate_free(out) };
        }
        unsafe { innovate_free(ptr::null_mut()) };
    }
}

/// Convert scenario execution rows from a grid response to Arrow IPC file bytes.
///
/// # Returns
/// Null on success, after writing the owned byte pointer to `data` and its byte
/// length to `len`. Release that original pair once with [`innovate_free_bytes`].
/// A non-null return instead points to an owned JSON error string that must be
/// released once with [`innovate_free`].
///
/// # Errors
/// Missing input or output pointers, invalid JSON and conversion failures return
/// `invalid_request`; a caught Rust panic returns `internal_error`. When both
/// output pointers are valid, they are initialized to null and zero before
/// conversion, including on failure. If either output pointer is null, no output
/// is initialized and neither out parameter should be read as a result.
///
/// # Safety
/// A non-null `response` must remain a readable, unchanged NUL-terminated string
/// for the call. Non-null `data` and `len` must point to properly aligned writable
/// storage for their declared types, disjoint from each other and from input
/// storage. Null pointers are rejected as request errors. Input remains borrowed;
/// the caller owns the responsibility for releasing either returned allocation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn innovate_scenario_executions_ipc(
    response: *const c_char,
    data: *mut *mut u8,
    len: *mut usize,
) -> *mut c_char {
    // SAFETY: forwarded caller contract.
    unsafe {
        scenario_output(
            response,
            data,
            len,
            innovate_tabular::scenario_executions_ipc,
        )
    }
}

/// Convert ordered grid comparisons to Arrow IPC.
///
/// The success/error return and byte-length out parameters follow
/// [`innovate_scenario_executions_ipc`]. Preserve the returned comparison order
/// when interpreting the output table.
///
/// # Errors
/// Missing pointers, invalid JSON and rejected comparison data return
/// `invalid_request`; a caught Rust panic returns `internal_error`.
///
/// # Safety
/// The pointer, lifetime and allocation rules of [`innovate_scenario_executions_ipc`]
/// apply, including disjoint writable output storage and exactly one matching free.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn innovate_scenario_comparisons_ipc(
    response: *const c_char,
    data: *mut *mut u8,
    len: *mut usize,
) -> *mut c_char {
    // SAFETY: forwarded caller contract.
    unsafe {
        scenario_output(
            response,
            data,
            len,
            innovate_tabular::scenario_comparisons_ipc,
        )
    }
}

/// Convert validated paired Bass outcomes to Arrow IPC.
///
/// `payload` is the paired-outcomes JSON object. The success/error return and
/// byte-length out parameters follow [`innovate_scenario_executions_ipc`].
///
/// # Errors
/// Missing pointers, invalid JSON or paired-outcome validation/conversion failures
/// return `invalid_request`; a caught Rust panic returns `internal_error`.
///
/// # Safety
/// The pointer, lifetime and allocation rules of [`innovate_scenario_executions_ipc`]
/// apply with `payload` as the borrowed string input.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn innovate_paired_outcomes_ipc(
    payload: *const c_char,
    data: *mut *mut u8,
    len: *mut usize,
) -> *mut c_char {
    // SAFETY: forwarded caller contract.
    unsafe { scenario_output(payload, data, len, innovate_tabular::paired_outcomes_ipc) }
}

unsafe fn scenario_output<E: std::fmt::Display>(
    response: *const c_char,
    data: *mut *mut u8,
    len: *mut usize,
    convert: fn(&serde_json::Value) -> Result<Vec<u8>, E>,
) -> *mut c_char {
    // SAFETY: caller supplies the response and writable out parameters.
    unsafe {
        arrow_output(data, len, || {
            if response.is_null() {
                return Err("missing Arrow export input".into());
            }
            let response = CStr::from_ptr(response);
            let value =
                serde_json::from_slice(response.to_bytes()).map_err(|err| err.to_string())?;
            convert(&value).map_err(|err| err.to_string())
        })
    }
}

/// Rewrite IPC file or stream bytes to a portable Arrow IPC file.
///
/// `input_len` and the returned `len` are byte counts. The success/error return,
/// output initialization and matching free functions follow
/// [`innovate_scenario_executions_ipc`]. Conversion allocates a new output buffer.
///
/// # Errors
/// Null pointers, an empty or oversized input, and IPC reader/writer failures
/// return `invalid_request`; a caught Rust panic returns `internal_error`.
///
/// # Safety
/// A non-null input must address `input_len` readable bytes in one allocation and
/// remain unchanged for the call. Non-null output pointers must be properly
/// aligned, writable for their types, and disjoint from each other and the input.
/// Release successful bytes with [`innovate_free_bytes`] and an error string with
/// [`innovate_free`], each exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn innovate_export_ipc_file(
    input: *const u8,
    input_len: usize,
    data: *mut *mut u8,
    len: *mut usize,
) -> *mut c_char {
    // SAFETY: caller supplies input bytes and writable out parameters.
    unsafe {
        arrow_output(data, len, || {
            if input.is_null() || input_len == 0 || input_len > isize::MAX as usize {
                return Err("missing or invalid IPC input bytes".into());
            }
            let bytes = std::slice::from_raw_parts(input, input_len);
            innovate_tabular::export_ipc_file(bytes).map_err(|err| err.to_string())
        })
    }
}

/// Rewrite IPC file or stream bytes to a portable Arrow IPC stream.
///
/// `input_len` and the returned `len` are byte counts. The success/error return,
/// output initialization and matching free functions follow
/// [`innovate_scenario_executions_ipc`]. Conversion allocates a new output buffer.
///
/// # Errors
/// Null pointers, an empty or oversized input, and IPC reader/writer failures
/// return `invalid_request`; a caught Rust panic returns `internal_error`.
///
/// # Safety
/// The borrowed-buffer, writable-output and allocation rules of
/// [`innovate_export_ipc_file`] apply. Input storage must remain allocated and
/// unchanged for the call; return either output allocation to its matching free
/// function exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn innovate_export_ipc_stream(
    input: *const u8,
    input_len: usize,
    data: *mut *mut u8,
    len: *mut usize,
) -> *mut c_char {
    // SAFETY: caller supplies input bytes and writable out parameters.
    unsafe {
        arrow_output(data, len, || {
            if input.is_null() || input_len == 0 || input_len > isize::MAX as usize {
                return Err("missing or invalid IPC input bytes".into());
            }
            let bytes = std::slice::from_raw_parts(input, input_len);
            innovate_tabular::export_ipc_stream(bytes).map_err(|err| err.to_string())
        })
    }
}

unsafe fn arrow_output(
    data: *mut *mut u8,
    len: *mut usize,
    convert: impl FnOnce() -> Result<Vec<u8>, String> + std::panic::UnwindSafe,
) -> *mut c_char {
    let error = |code: &str, message: &str| {
        owned_response(Ok(
            serde_json::json!({"error":{"code":code,"message":message}}).to_string(),
        ))
    };
    if data.is_null() || len.is_null() {
        return error("invalid_request", "missing Arrow output pointers");
    }
    // SAFETY: out parameters are writable for the duration of the call.
    unsafe {
        *data = ptr::null_mut();
        *len = 0;
    }
    let result = std::panic::catch_unwind(convert);
    match result {
        Ok(Ok(bytes)) => {
            let bytes = bytes.into_boxed_slice();
            // SAFETY: caller provides writable out parameters.
            unsafe {
                *len = bytes.len();
                *data = Box::into_raw(bytes).cast();
            }
            ptr::null_mut()
        }
        Ok(Err(message)) => error("invalid_request", &message),
        Err(_) => error("internal_error", "native execution panicked"),
    }
}

/// Release an Arrow result once using its original byte length.
///
/// A null `data` pointer is a no-op. JSON error strings use [`innovate_free`]
/// instead; no output can be read after its allocation has been released.
///
/// # Safety
/// A non-null pointer and length must be the original pair returned by
/// an Arrow export, and must still be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn innovate_free_bytes(data: *mut u8, len: usize) {
    if !data.is_null() {
        // SAFETY: caller returns the original allocation and length once.
        drop(unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(data, len)) });
    }
}
