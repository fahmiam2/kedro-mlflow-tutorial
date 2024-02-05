import pandas as pd

def drop_unnecessary_columns(raw_data: pd.DataFrame, columns: list) -> pd.DataFrame:
    cleaned_data = raw_data.copy()
    return cleaned_data.drop(columns, axis=1)