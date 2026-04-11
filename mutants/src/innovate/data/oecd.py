import pandasdmx as sdmx
from typing import Annotated
from typing import Callable
from typing import ClassVar

MutantDict = Annotated[dict[str, Callable], "Mutant"] # type: ignore


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg = None): # type: ignore
    """Forward call to original or mutated function, depending on the environment"""
    import os # type: ignore
    mutant_under_test = os.environ['MUTANT_UNDER_TEST'] # type: ignore
    if mutant_under_test == 'fail': # type: ignore
        from mutmut.__main__ import MutmutProgrammaticFailException # type: ignore
        raise MutmutProgrammaticFailException('Failed programmatically')       # type: ignore
    elif mutant_under_test == 'stats': # type: ignore
        from mutmut.__main__ import record_trampoline_hit # type: ignore
        record_trampoline_hit(orig.__module__ + '.' + orig.__name__) # type: ignore
        # (for class methods, orig is bound and thus does not need the explicit self argument)
        result = orig(*call_args, **call_kwargs) # type: ignore
        return result # type: ignore
    prefix = orig.__module__ + '.' + orig.__name__ + '__mutmut_' # type: ignore
    if not mutant_under_test.startswith(prefix): # type: ignore
        result = orig(*call_args, **call_kwargs) # type: ignore
        return result # type: ignore
    mutant_name = mutant_under_test.rpartition('.')[-1] # type: ignore
    if self_arg is not None: # type: ignore
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs) # type: ignore
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs) # type: ignore
    return result # type: ignore


def get_dataset(dataset_id, dimensions, start_date="1960", end_date="2020"):
    args = [dataset_id, dimensions, start_date, end_date]# type: ignore
    kwargs = {}# type: ignore
    return _mutmut_trampoline(x_get_dataset__mutmut_orig, x_get_dataset__mutmut_mutants, args, kwargs, None)


def x_get_dataset__mutmut_orig(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request("OECD")
    data_msg = oecd.data(
        resource_id=dataset_id,
        key=dimensions,
        params={"startTime": start_date, "endTime": end_date},
    )
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_1(dataset_id, dimensions, start_date="XX1960XX", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request("OECD")
    data_msg = oecd.data(
        resource_id=dataset_id,
        key=dimensions,
        params={"startTime": start_date, "endTime": end_date},
    )
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_2(dataset_id, dimensions, start_date="1960", end_date="XX2020XX"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request("OECD")
    data_msg = oecd.data(
        resource_id=dataset_id,
        key=dimensions,
        params={"startTime": start_date, "endTime": end_date},
    )
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_3(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = None
    data_msg = oecd.data(
        resource_id=dataset_id,
        key=dimensions,
        params={"startTime": start_date, "endTime": end_date},
    )
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_4(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request(None)
    data_msg = oecd.data(
        resource_id=dataset_id,
        key=dimensions,
        params={"startTime": start_date, "endTime": end_date},
    )
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_5(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request("XXOECDXX")
    data_msg = oecd.data(
        resource_id=dataset_id,
        key=dimensions,
        params={"startTime": start_date, "endTime": end_date},
    )
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_6(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request("oecd")
    data_msg = oecd.data(
        resource_id=dataset_id,
        key=dimensions,
        params={"startTime": start_date, "endTime": end_date},
    )
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_7(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request("OECD")
    data_msg = None
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_8(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request("OECD")
    data_msg = oecd.data(
        resource_id=None,
        key=dimensions,
        params={"startTime": start_date, "endTime": end_date},
    )
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_9(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request("OECD")
    data_msg = oecd.data(
        resource_id=dataset_id,
        key=None,
        params={"startTime": start_date, "endTime": end_date},
    )
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_10(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request("OECD")
    data_msg = oecd.data(
        resource_id=dataset_id,
        key=dimensions,
        params=None,
    )
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_11(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request("OECD")
    data_msg = oecd.data(
        key=dimensions,
        params={"startTime": start_date, "endTime": end_date},
    )
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_12(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request("OECD")
    data_msg = oecd.data(
        resource_id=dataset_id,
        params={"startTime": start_date, "endTime": end_date},
    )
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_13(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request("OECD")
    data_msg = oecd.data(
        resource_id=dataset_id,
        key=dimensions,
        )
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_14(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request("OECD")
    data_msg = oecd.data(
        resource_id=dataset_id,
        key=dimensions,
        params={"XXstartTimeXX": start_date, "endTime": end_date},
    )
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_15(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request("OECD")
    data_msg = oecd.data(
        resource_id=dataset_id,
        key=dimensions,
        params={"starttime": start_date, "endTime": end_date},
    )
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_16(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request("OECD")
    data_msg = oecd.data(
        resource_id=dataset_id,
        key=dimensions,
        params={"STARTTIME": start_date, "endTime": end_date},
    )
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_17(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request("OECD")
    data_msg = oecd.data(
        resource_id=dataset_id,
        key=dimensions,
        params={"startTime": start_date, "XXendTimeXX": end_date},
    )
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_18(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request("OECD")
    data_msg = oecd.data(
        resource_id=dataset_id,
        key=dimensions,
        params={"startTime": start_date, "endtime": end_date},
    )
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_19(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request("OECD")
    data_msg = oecd.data(
        resource_id=dataset_id,
        key=dimensions,
        params={"startTime": start_date, "ENDTIME": end_date},
    )
    data = data_msg.to_pandas()
    return data


def x_get_dataset__mutmut_20(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """Gets a dataset from the OECD."""
    oecd = sdmx.Request("OECD")
    data_msg = oecd.data(
        resource_id=dataset_id,
        key=dimensions,
        params={"startTime": start_date, "endTime": end_date},
    )
    data = None
    return data

x_get_dataset__mutmut_mutants : ClassVar[MutantDict] = { # type: ignore
'x_get_dataset__mutmut_1': x_get_dataset__mutmut_1, 
    'x_get_dataset__mutmut_2': x_get_dataset__mutmut_2, 
    'x_get_dataset__mutmut_3': x_get_dataset__mutmut_3, 
    'x_get_dataset__mutmut_4': x_get_dataset__mutmut_4, 
    'x_get_dataset__mutmut_5': x_get_dataset__mutmut_5, 
    'x_get_dataset__mutmut_6': x_get_dataset__mutmut_6, 
    'x_get_dataset__mutmut_7': x_get_dataset__mutmut_7, 
    'x_get_dataset__mutmut_8': x_get_dataset__mutmut_8, 
    'x_get_dataset__mutmut_9': x_get_dataset__mutmut_9, 
    'x_get_dataset__mutmut_10': x_get_dataset__mutmut_10, 
    'x_get_dataset__mutmut_11': x_get_dataset__mutmut_11, 
    'x_get_dataset__mutmut_12': x_get_dataset__mutmut_12, 
    'x_get_dataset__mutmut_13': x_get_dataset__mutmut_13, 
    'x_get_dataset__mutmut_14': x_get_dataset__mutmut_14, 
    'x_get_dataset__mutmut_15': x_get_dataset__mutmut_15, 
    'x_get_dataset__mutmut_16': x_get_dataset__mutmut_16, 
    'x_get_dataset__mutmut_17': x_get_dataset__mutmut_17, 
    'x_get_dataset__mutmut_18': x_get_dataset__mutmut_18, 
    'x_get_dataset__mutmut_19': x_get_dataset__mutmut_19, 
    'x_get_dataset__mutmut_20': x_get_dataset__mutmut_20
}
x_get_dataset__mutmut_orig.__name__ = 'x_get_dataset'
