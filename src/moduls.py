import pandas as pd
import numpy as np
np.random.seed(42)

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

def healthy_vs_diseased(df):
    mask_disease = df["disease"] > 0
    mask_healthy = df["disease"] < 1

    df_disease = df[mask_disease]
    df_healthy = df[mask_healthy]

    return df_disease, df_healthy

def frequency_of_diseased(df):
    tu = healthy_vs_diseased(df)
    tu[0].value_counts().sum()
    tu[1].value_counts().sum()

    return round((tu[0].value_counts().sum() / (tu[0].value_counts().sum() + tu[1].value_counts().sum())), 3)

def creating_a_random_data_set_with_disease_frequency(df):
    outcome = [0, 1]
    disease_probability = [1 - frequency_of_diseased(df), frequency_of_diseased(df)]
    n = 1000
    data = np.random.choice(a=outcome, size=n, p=disease_probability)
    return data

def ci_mean_norma(x, confidence=0.95):

    x = np.asarray(x, dtype=float)
    mean_x = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    n = len(x)

    z_critical = 1.96

    margin_of_error = z_critical * (s / np.sqrt(n))

    lo, hi = mean_x - margin_of_error, mean_x + margin_of_error

    return lo, hi, mean_x, s, n

def ci_mean_bootstrap(x, B=10000, confidence=0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    boot_means = np.empty(B)
    for i in range(B):
        boot_sample = np.random.choice(x, size=n, replace=True)
        boot_means[i] = np.mean(boot_sample)

    alpha = (1 - confidence) / 2
    blo, bhi = np.percentile(boot_means, [100*alpha, 100*(1 - alpha)])
    return float(blo), float(bhi), float(np.mean(x))
    
def looking_for_them_smokers(df: pd.DataFrame):
    mask_smokers = df["smoker"] == "Yes"
    mask_non_smokers = df["smoker"] == "No"

    df_smokers = df[mask_smokers]
    df_non_smokers = df[mask_non_smokers]

    return df_smokers, df_non_smokers

