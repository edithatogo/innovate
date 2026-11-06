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
    let text = response.unwrap_or_else(|_| {
        r#"{"error":{"code":"internal_error","message":"native execution panicked"},"result":null}"#.into()
    });
    CString::new(text).map_or(ptr::null_mut(), CString::into_raw)
}

/// Release a result returned by `innovate_call`. A null pointer is a no-op.
///
/// # Safety
/// `response` must be null or a still-live pointer returned by `innovate_call`.
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
