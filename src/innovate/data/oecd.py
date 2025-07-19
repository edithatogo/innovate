import pandas as pd
import pandasdmx as sdmx

def get_dataset(dataset_id, dimensions, start_date="1960", end_date="2020"):
    """
    Gets a dataset from the OECD.
    """
    oecd = sdmx.Request('OECD')
    data_msg = oecd.data(resource_id=dataset_id, key=dimensions, params={'startTime': start_date, 'endTime': end_date})
    data = data_msg.to_pandas()
    return data
