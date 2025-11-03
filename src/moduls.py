import pandas as pd
import numpy as np

def showing_standard_info(df: pd.DataFrame) -> pd.DataFrame:
    list_of_some_numeric_categories = ["age", "weight", "height", "systolic_bp", "cholesterol"]
    dictionary_max = {}
    dictionary_min = {}
    dictionary_mean =  {}
    dictionary_median = {}

    for category in list_of_some_numeric_categories:
        dictionary_max[category] = df[category].max()
        dictionary_min[category] = df[category].min()
        dictionary_mean[category] = df[category].mean()
        dictionary_median[category] = df[category].median()

    series_max = pd.Series(dictionary_max, name="Max")
    series_min = pd.Series(dictionary_min, name="Min")
    series_mean = pd.Series(dictionary_mean, name="Medel")
    series_median = pd.Series(dictionary_median, name="Median")

    el_finalo_dataframe = pd.DataFrame([series_max, series_min, series_mean, series_median])
    return el_finalo_dataframe

def healthy_vs_diseased_info(df):
    mask_disease = df["disease"] > 0
    mask_healthy = df["disease"] < 1

    df_disease = df[mask_disease]
    df_healthy = df[mask_healthy]

    number_of_disease = mask_disease.sum()
    number_of_healthy = mask_healthy.sum()

    df1 = df_disease.groupby(["smoker", "sex"])["systolic_bp"].mean()
    df2 = df_healthy.groupby(["smoker", "sex"])["systolic_bp"].mean()
    
    return df1, df2

