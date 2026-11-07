//! Narrow C ABI for Julia and R. Input is borrowed; returned UTF-8 is Rust-owned.
use std::ffi::{CStr, CString, c_char};
use std::ptr;

/// Execute a UTF-8 JSON request. Free the result once with `innovate_free`.
/// Null input returns a structured invalid-request response.
///
/// # Safety
/// Non-null `request` must point to a readable NUL-terminated string for this call.
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

/// Execute a scenario grid on a borrowed Arrow IPC file. Free with innovate_free.
///
/// # Safety
/// Non-null pointers must address a readable buffer of length `len` and a
/// readable NUL-terminated payload respectively for the duration of the call.
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

/// Release a result returned by a native call. A null pointer is a no-op.
///
/// # Safety
/// `response` must be null or a still-live pointer returned by a native call.
/// Do not use the pointer after this call, free it twice, or supply foreign memory.
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

/// Convert a grid response to Arrow IPC. Returns a JSON error or null on success.
///
/// # Safety
/// The response must be a readable NUL-terminated string. Both out parameters
/// must be valid writable pointers. Release successful bytes with
/// `innovate_free_bytes` and any error string with `innovate_free`.
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
/// # Safety
/// Same pointer and ownership contract as `innovate_scenario_executions_ipc`.
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
                return Err("missing scenario grid response".into());
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
/// # Safety
/// Input must be readable for input_len bytes. Output pointers must be writable.
/// Release output using innovate_free_bytes or innovate_free for an error.
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

/// Release an Arrow result once.
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
