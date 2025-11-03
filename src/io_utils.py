import pandas as pd

# Opening the csv file and returning a dataframe
def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

# Cleaning the dataframe
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)

    # Because all the data seems to be in order, no further cleaning needs to be done
    return df

    

