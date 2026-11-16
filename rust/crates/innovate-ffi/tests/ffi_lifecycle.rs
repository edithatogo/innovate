use innovate_ffi::{innovate_call, innovate_free, innovate_scenario_grid_ipc};
use std::ffi::{CStr, CString};
use std::ptr;

#[test]
fn ffi_null_malformed_and_repeated_calls_are_safe_and_leak_free() {
    // Null free is a safe no-op.
    unsafe { innovate_free(ptr::null_mut()) };

    // Null request returns structured invalid_request JSON.
    let null_out = unsafe { innovate_call(ptr::null()) };
    assert!(!null_out.is_null());
    let parsed: serde_json::Value =
        serde_json::from_slice(unsafe { CStr::from_ptr(null_out) }.to_bytes()).unwrap();
    unsafe { innovate_free(null_out) };
    assert!(!parsed["error"].is_null());

    // Repeated valid calls succeed and clean up cleanly.
    let req = CString::new(
        serde_json::json!({
            "schema_version": "2.0",
            "operation": "predict_model",
            "model_key": "bass",
            "payload": {
                "inputs": {"time": [0.0, 1.0, 2.0]},
                "parameters": {"p": 0.03, "q": 0.38, "m": 1000.0}
            }
        })
        .to_string(),
    )
    .unwrap();
    for _ in 0..10 {
        let out = unsafe { innovate_call(req.as_ptr()) };
        assert!(!out.is_null());
        let resp: serde_json::Value =
            serde_json::from_slice(unsafe { CStr::from_ptr(out) }.to_bytes()).unwrap();
        unsafe { innovate_free(out) };
        assert!(resp["error"].is_null());
        assert_eq!(resp["result"]["values"].as_array().unwrap().len(), 3);
    }

    // Invalid Arrow IPC pointer/length fails closed without panic.
    let payload = CString::new("{}").unwrap();
    let ipc_out = unsafe { innovate_scenario_grid_ipc(ptr::null(), 0, payload.as_ptr()) };
    assert!(!ipc_out.is_null());
    let ipc_resp: serde_json::Value =
        serde_json::from_slice(unsafe { CStr::from_ptr(ipc_out) }.to_bytes()).unwrap();
    unsafe { innovate_free(ipc_out) };
    assert_eq!(ipc_resp["error"]["code"], "invalid_request");
}
