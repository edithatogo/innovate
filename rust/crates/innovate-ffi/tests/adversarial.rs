use innovate_ffi::{innovate_call, innovate_free, innovate_scenario_grid_ipc};
use std::ffi::{CStr, CString};
use std::ptr;

#[test]
fn ffi_survives_adversarial_inputs_without_panics_or_leaks() {
    let invalid_utf8 = CString::from_vec_with_nul(vec![0xff, 0xfe, 0x00]).unwrap();
    let empty_cstr = CString::new("").unwrap();
    let malformed = CString::new("{broken json").unwrap();
    let huge = CString::new("x".repeat(64 * 1024)).unwrap();

    for ptr_input in [
        ptr::null(),
        invalid_utf8.as_ptr(),
        empty_cstr.as_ptr(),
        malformed.as_ptr(),
        huge.as_ptr(),
    ] {
        let out = unsafe { innovate_call(ptr_input) };
        assert!(!out.is_null());
        let resp: serde_json::Value =
            serde_json::from_slice(unsafe { CStr::from_ptr(out) }.to_bytes()).unwrap();
        unsafe { innovate_free(out) };
        assert!(!resp["error"].is_null());
    }

    let valid_payload = CString::new("{}").unwrap();
    let bogus_bytes = [1u8, 2, 3, 4];
    let ipc_err = unsafe {
        innovate_scenario_grid_ipc(
            bogus_bytes.as_ptr(),
            bogus_bytes.len(),
            valid_payload.as_ptr(),
        )
    };
    assert!(!ipc_err.is_null());
    let resp: serde_json::Value =
        serde_json::from_slice(unsafe { CStr::from_ptr(ipc_err) }.to_bytes()).unwrap();
    unsafe { innovate_free(ipc_err) };
    assert_eq!(resp["error"]["code"], "invalid_request");
}
