import pandas as pd
import wbdata

def get_indicator(indicator, country="all", start_date="1960", end_date="2020"):
    """
    Gets data for a given indicator from the World Bank.
    """
    data_date = pd.to_datetime(f"{start_date}-01-01"), pd.to_datetime(f"{end_date}-12-31")
    df = wbdata.get_dataframe({indicator: "value"}, country=country, data_date=data_date, convert_date=True)
    return df
