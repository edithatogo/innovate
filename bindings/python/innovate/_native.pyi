"""Typed transport surface of Innovate's compiled PyO3 extension.

Prefer the public ``innovate`` wrappers for Polars frames and structured native
errors. These transport functions accept serialized JSON and Arrow IPC, release
the Python interpreter while executing native work, and copy IPC buffers. Native
tabular failures become ``ValueError``; kernel failures returned in JSON envelopes
must be decoded and checked by the caller.

Attributes:
    __version__: Version compiled into the native crate.
"""

def execute(request: str) -> str:
    """Execute a serialized native request without decoding its response.

    Args:
        request: JSON request envelope with native schema, operation and payload.

    Returns:
        JSON response envelope. Invalid requests and scientific failures are
        represented in its ``error`` field rather than raised by this function.

    Raises:
        TypeError: If the argument cannot be extracted as a Python string.
    """
    ...

def export_ipc_file(data: bytes) -> bytes:
    """Re-encode Arrow IPC file or stream bytes with an interoperable file writer.

    Args:
        data: Complete Arrow IPC file or stream bytes to decode natively.

    Returns:
        Newly allocated IPC file bytes, retaining schema metadata and compatible
        field annotations from the input. The conversion copies buffers.

    Raises:
        ValueError: If the native reader or writer rejects the input.
        TypeError: If input cannot be extracted as the native byte-vector argument.
    """
    ...

def export_ipc_stream(data: bytes) -> bytes:
    """Re-encode Arrow IPC bytes as a stream carrying table schema metadata.

    Args:
        data: Complete IPC file or stream bytes, decoded by native Polars.

    Returns:
        Newly allocated stream bytes with compatible metadata retained. This
        serialized transport is not a zero-copy Arrow C Data interface.

    Raises:
        ValueError: If decoding or writing the native IPC stream fails.
        TypeError: If input cannot be extracted as the native byte-vector argument.
    """
    ...

def predict_ipc(data: bytes, model: str, parameters: str, options: str | None = None) -> bytes:
    """Append native prediction columns to a table transported as Arrow IPC.

    Args:
        data: IPC table with finite, non-null, strictly increasing numeric ``time``
            and any covariate columns required by the model options.
        model: Native model key selecting the prediction implementation.
        parameters: JSON mapping of finite named model parameters.
        options: Optional JSON mapping of ``model_options``, ``quantity`` and
            ``covariate_rate_columns``. ``None`` selects an empty option mapping.

    Returns:
        IPC file bytes retaining original columns and order, with ``prediction``
        or ``rate`` output columns, numbered for multi-series model outputs.

    Raises:
        ValueError: If JSON, table schema, native prediction or output validation fails.
        TypeError: If Python arguments cannot be extracted as their declared types.
    """
    ...

def fit_ipc(data: bytes, model: str, options: str | None = None) -> str:
    """Fit a native model from an IPC observation table and return its envelope.

    Args:
        data: IPC table with finite increasing ``time`` and non-null numeric
            observed columns. The default observation column is ``observed``.
        model: Native model key selecting the fitting implementation.
        options: Optional JSON mapping of ``model_options``, ``observed_columns``
            and ``fit_options``; ``None`` supplies an empty mapping.

    Returns:
        JSON fit envelope. Scientific fitting errors may be present in ``error``
        and must be checked after decoding; no Python FitResult is created here.

    Raises:
        ValueError: If JSON options or the IPC table fail transport/schema validation.
        TypeError: If arguments cannot be extracted as the declared native types.
    """
    ...

def categorize_adopters_ipc(data: bytes, rate_column: str) -> bytes:
    """Append historical sample-weighted adopter categories to an IPC table.

    Args:
        data: IPC table containing strictly increasing numeric ``time`` and finite
            nonnegative rates with positive total mass, all without nulls.
        rate_column: Existing numeric column containing the sampled adoption rates.

    Returns:
        IPC file bytes with original columns and an added ``category`` column.

    Raises:
        ValueError: If rate/time validation, category-name uniqueness or IPC conversion fails.
        TypeError: If arguments cannot be extracted as the declared native types.
    """
    ...

def scenario_grid_ipc(data: bytes, payload: str) -> str:
    """Inject a shared IPC time grid and execute a native scenario request.

    Args:
        data: IPC table with a finite, non-null, strictly increasing ``time`` column.
        payload: JSON scenario-grid mapping. Each scenario supplies a simulation
            object whose inputs omit both ``time`` and ``t``.

    Returns:
        JSON scenario-grid response envelope; callers must inspect its outer and
        per-scenario errors. Time values are copied into each simulation payload.

    Raises:
        ValueError: If IPC, JSON, the shared grid or scenario input structure is invalid.
        TypeError: If arguments cannot be extracted as the declared native types.
    """
    ...

def scenario_executions_ipc(response: str) -> bytes:
    """Encode ordered scenario execution rows from a successful grid envelope.

    Args:
        response: JSON ``execute_scenario_grid`` response with no outer error.

    Returns:
        IPC file bytes with scenario IDs, model keys, success flags and complete
        nested execution JSON. Paused execution rows are not marked successful.

    Raises:
        ValueError: If the envelope, execution rows or IPC encoding are invalid.
        TypeError: If the response cannot be extracted as a Python string.
    """
    ...

def scenario_comparisons_ipc(response: str) -> bytes:
    """Encode ordered pair and ensemble comparisons from a grid envelope.

    Args:
        response: JSON successful grid envelope with a ``result.comparisons`` array.

    Returns:
        IPC bytes with comparison indices, kind, success flags and complete
        nested comparison JSON. Individual comparison errors remain represented.

    Raises:
        ValueError: If JSON, the envelope, comparison rows or IPC encoding is invalid.
        TypeError: If the response cannot be extracted as a Python string.
    """
    ...

def paired_outcomes_ipc(payload: str) -> bytes:
    """Validate paired Bass trajectories and encode ordered outcome rows as IPC.

    Args:
        payload: JSON native paired-outcomes contract containing ordered strategies,
            paired draw IDs, parameters, time grids and adoption trajectories.

    Returns:
        IPC file bytes in strategy, draw and time order, retaining provenance,
        units, optional perspective and original named parameter records as JSON.

    Raises:
        ValueError: If JSON, pairing, native trajectory validation or IPC encoding fails.
        TypeError: If payload cannot be extracted as a Python string.
    """
    ...

__version__: str
