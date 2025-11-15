import pandas as pd
import numpy as np
from scipy import stats
from scipy import linalg

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
    series_mean = pd.Series(dictionary_mean, name="Mean")
    series_median = pd.Series(dictionary_median, name="Median")

    el_finalo_dataframe = pd.DataFrame([series_max, series_min, series_mean, series_median])
    return el_finalo_dataframe

def healthy_vs_diseased_info(df):
    mask_disease = df["disease"] > 0
    mask_healthy = df["disease"] < 1

    df_disease = df[mask_disease]
    df_healthy = df[mask_healthy]

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
    np.random.seed(42)
    outcome = [0, 1]
    disease_probability = [1 - frequency_of_diseased(df), frequency_of_diseased(df)]
    n = 1000
    data = np.random.choice(a=outcome, size=n, p=disease_probability)
    return data

def actual_frequency_vs_random_generated_frequency(df: pd.DataFrame):
    np.random.seed(42)
    actual_frequency = frequency_of_diseased(df)
    random_dataset = creating_a_random_data_set_with_disease_frequency(df)
    random_frequency = round(np.mean(random_dataset), 3)

    dictionary_of_disease_frequency_and_frequency_from_random_dataset = {}
    dictionary_of_disease_frequency_and_frequency_from_random_dataset["Actual disease frequency"] = actual_frequency
    dictionary_of_disease_frequency_and_frequency_from_random_dataset["Random dataset disease frequency"] = random_frequency
    dictionary_of_disease_frequency_and_frequency_from_random_dataset["Difference between actual and random frequency"] = round(abs(actual_frequency - random_frequency), 3)

    df_frequency = pd.Series(dictionary_of_disease_frequency_and_frequency_from_random_dataset)
    return df_frequency


def looking_for_them_smokers(df: pd.DataFrame):
    mask_smokers = df["smoker"] == "Yes"
    mask_non_smokers = df["smoker"] == "No"

    df_smokers = df[mask_smokers]
    df_non_smokers = df[mask_non_smokers]

    return df_smokers, df_non_smokers


class CI_and_bootstrap:
    def __init__(self, data, confidence=0.95):

        self.data = np.asarray(data, dtype=float)
        self.confidence = confidence
        self.n = len(self.data)
        
    def ci_mean_norma(self):

        mean_x = float(np.mean(self.data))
        s = float(np.std(self.data, ddof=1))
        
        z_critical = 1.96  # For 95% confidence
        
        margin_of_error = z_critical * (s / np.sqrt(self.n))
        
        lo = mean_x - margin_of_error
        hi = mean_x + margin_of_error
        
        return lo, hi, mean_x, s, self.n
    
    def ci_mean_bootstrap(self, B=10000):

        boot_means = np.empty(B)
        for i in range(B):
            boot_sample = np.random.choice(self.data, size=self.n, replace=True)
            boot_means[i] = np.mean(boot_sample)
        
        alpha = (1 - self.confidence) / 2
        blo, bhi = np.percentile(boot_means, [100*alpha, 100*(1 - alpha)])
        
        return float(blo), float(bhi), float(np.mean(self.data))
    
    def compare_methods(self, B=10000):

        lo_norm, hi_norm, mean_norm, s, n = self.ci_mean_norma()
        lo_boot, hi_boot, mean_boot = self.ci_mean_bootstrap(B=B)
        
        comparison_df = pd.DataFrame({
            'Method': ['Normalapproximation', 'Bootstrap'],
            'Lower limit': [lo_norm, lo_boot],
            'Upper limit': [hi_norm, hi_boot],
            'Mean': [mean_norm, mean_boot],
            'Interval width': [hi_norm - lo_norm, hi_boot - lo_boot]
        })
        
        differences = {
            'lower_limit': abs(lo_norm - lo_boot),
            'upper_limit': abs(hi_norm - hi_boot),
            'mean': abs(mean_norm - mean_boot)
        }
        
        print("=== JÄMFÖRELSE AV KONFIDENSINTERVALL ===\n")
        print(comparison_df)
        print("\n=== SKILLNADER MELLAN METODERNA ===")
        print(f"Difference in lower limit: {differences['lower_limit']:.4f}")
        print(f"Difference in upper limit: {differences['upper_limit']:.4f}")
        print(f"Difference in mean: {differences['mean']:.4f}")
        
        return {
            'comparison_df': comparison_df,
            'differences': differences,
            'normal': {'lo': lo_norm, 'hi': hi_norm, 'mean': mean_norm},
            'bootstrap': {'lo': lo_boot, 'hi': hi_boot, 'mean': mean_boot}
        }
    


def checking_power_of_that_t_test(n_experiment_group,
                                  n_control_group,
                                  std_experiment_group,
                                  std_control_group,
                                  diff_to_find,
                                  alpha = 0.05,
                                  n_simulations = 1000,
                                  alternative = "greater"
                                  ):
    
    mean_control = 120    
    mean_experiment = mean_control + diff_to_find
    significant_results = 0
    np.random.seed(42)
    for _ in range(n_simulations):

        experiment_data = np.random.normal(loc=mean_experiment, scale=std_experiment_group, size=n_experiment_group)
        control_data = np.random.normal(loc=mean_control, scale=std_control_group, size=n_control_group)
        if alternative == "greater":
            t_stat, p_value = stats.ttest_ind(
                a=experiment_data, 
                b=control_data, 
                equal_var=False,
                alternative='greater'
            )
        elif alternative == "less":
                 t_stat, p_value = stats.ttest_ind(
                a=experiment_data, 
                b=control_data, 
                equal_var=False,
                alternative='less'
            )
        else:
            t_stat, p_value = stats.ttest_ind(
                a=experiment_data, 
                b=control_data, 
                equal_var=False,
                alternative="two-sided"
            )

        if p_value < alpha:
            significant_results += 1

    power = significant_results / n_simulations

    print(f"--- Simulation Results ---")
    print(f"Assumed true difference (Effect size): {diff_to_find} mmHg")
    print(f"Sample sizes: Sick={n_experiment_group}, Healthy={n_control_group}")
    print(f"Significance level (Alpha): {alpha}")
    print(f"\nNumber of significant results: {significant_results} of {n_simulations}")
    print(f"Calculated power (Power): {power:.3f}")
    print("\n")
    return power, diff_to_find
