🧪 [testing improvement] Add unit tests for update_streaming_forecast

🎯 **What:**
Added unit tests for the `update_streaming_forecast` function in `src/innovate/advanced_runtime.py`, closing a testing gap for this function.

📊 **Coverage:**
The new unit tests cover:
- Happy path streaming forecast update behavior and payload metadata/diagnostics mapping.
- Error conditions: mismatched lengths between previous time/observed vectors.
- Error conditions: mismatched lengths between new time/observed vectors.
- Error conditions: combined time points that are not sorted.
- Error conditions: combined observed values that are not strictly cumulative.

✨ **Result:**
Improved test coverage, ensuring `update_streaming_forecast` retains reliability by raising expected `ValueError`s correctly, and generating streaming update results correctly in a fast, independent unit test environment.
