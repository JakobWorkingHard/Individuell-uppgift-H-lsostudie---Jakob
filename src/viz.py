import matplotlib.pyplot as plt
from src.moduls import *


def plot_gender_weight_difference(df: pd.DataFrame):
    """
    This function plots a boxplot of column="weight" and by="sex", using only your dataframe as input
    """
    fig, ax = plt.subplots(figsize=(8,5))
    df.boxplot(column="weight", by="sex", ax=ax)
    ax.set_title("Weight for males/females")
    ax.set_xlabel("")
    ax.set_ylabel("weight in kilos")
    plt.suptitle("")
    plt.tight_layout()
    plt.show


def plot_disease_vs_healthy(df1, df2: pd.DataFrame):
    """
    Inputs: This function requires two dataframe inputs, one for non diseased peoples average blood pressure and one with disease-peoples average blood pressure

    How to obtain the dataframes: Use the function "healthy vs diseased info" to obtain a tuple of the two dataframes, and then input it in this plot.

    What this function does: This function plots 3 barplots comparing diseased smokers/non smokers vs non-diseased smokers/non smokers blood pressure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)

    # Plotting non diseased
    df1.plot(kind="bar", ax=axes[0], legend=False, color="green")
    axes[0].set_title("Non disease-people")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Mean - Blood pressure")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].set_ylim(140, 160)

    # Plotting diseased
    df2.plot(kind="bar", ax=axes[1], legend=False, color="red")
    axes[1].set_title("Disease-people")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean - Blood pressure")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].set_ylim(140, 160)

    x = np.arange(len(df1))
    width = 0.35
    # Plotting them both next to each other's
    axes[2].bar(x - width/2, df1.values, width, label="Non disease", color="green", alpha=0.8)
    axes[2].bar(x + width/2, df2.values, width, label="Disease", color="red", alpha=0.8)
    
    axes[2].set_title("Comparison")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Mean - Blood pressure")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(df1.index, rotation=0)
    axes[2].set_ylim(140, 160)
    axes[2].legend()

    fig.suptitle("Comparing blood pressure levels of sick and healthy people with smoker/non smoker and sex")
    plt.tight_layout()
    plt.show()
    

def plot_comparing_age_with_cholesterol(df1, df2: pd.DataFrame):
    """
    Plotting three scatter plots, first comparing diseased cholesterol compared to age, second non diseased cholesterol compared to age, 
    and third both diseased and non diseased cholesterol compared to age with linear regression line for each dataset.
    
    Input: Two dataframes, one with diseased only, and one with non diseased only
    """
    fig = plt.figure(figsize=(10, 12))
    
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # First plot - Disease 
    ax0 = fig.add_subplot(gs[0, 0])
    df1.plot(kind="scatter", ax=ax0, legend=False, x="age", y="cholesterol", color="red")
    ax0.set_title("Comparing age with cholesterol - disease")
    ax0.set_ylabel("Cholesterol")
    ax0.tick_params(axis="x", rotation=0)
    ax0.set_ylim(2, 8)
    ax0.set_xlim(0, 90)

    # Second plot - Non diseased
    ax1 = fig.add_subplot(gs[0, 1])
    df2.plot(kind="scatter", ax=ax1, legend=False, x="age", y="cholesterol", color="green")
    ax1.set_title("Comparing age with cholesterol - healthy")
    ax1.set_ylabel("Cholesterol")
    ax1.tick_params(axis="x", rotation=0)
    ax1.set_ylim(2, 8)
    ax1.set_xlim(0, 90)

    # Third plot - Combined + regression lines 
    ax2 = fig.add_subplot(gs[1, :])  
    ax2.scatter(df1["age"], df1["cholesterol"], color="red", alpha=0.5, label="Disease", s=20)
    ax2.scatter(df2["age"], df2["cholesterol"], color="green", alpha=0.5, label="Healthy", s=20)
    
    z1 = np.polyfit(df1["age"], df1["cholesterol"], 1)
    p1 = np.poly1d(z1)
    ax2.plot(df1["age"].sort_values(), p1(df1["age"].sort_values()), 
             color="darkred", linewidth=2, linestyle="--", label="Disease trend")
    
    z2 = np.polyfit(df2["age"], df2["cholesterol"], 1)
    p2 = np.poly1d(z2)
    ax2.plot(df2["age"].sort_values(), p2(df2["age"].sort_values()), 
             color="darkgreen", linewidth=2, linestyle="--", label="Healthy trend")
    
    ax2.set_title("Combined comparison with regression lines")
    ax2.set_xlabel("age")
    ax2.set_ylabel("Cholesterol")
    ax2.set_ylim(2, 8)
    ax2.set_xlim(0, 90)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def just_a_basic_plot(df: pd.DataFrame, x_label_and_category: str, y_label_and_category: str):
    """
    Plots a simple barplot of your groupby:ed dataframe.
    """
    fig, ax = plt.subplot(figsize=(10,8))
    df.bar(x_label_and_category, y_label_and_category, color="seagreen")
    ax.set_title("")
    ax.set_xlabel(x_label_and_category)
    ax.set_ylabel(y_label_and_category)
    plt.tight_layout()
    plt.show()


def plot_actual_frequency_vs_random_generated_frequency(df):
    """
    Plots a barplot of three different bars of colors blue, yellow and red

    Input: Requeres a dataframe of 3 rows
    """
    df.plot.bar(color=["blue", "yellow", "red"])
    plt.ylim(0, 0.08)  
    plt.tick_params(axis="x", rotation=60)
    plt.tight_layout()
    plt.show()


def plot_ci_comparison(comparison_results: dict):
    """
    This plot visualizes the comparison of two confidence intervals, one with normal approximation and one with bootstrap.

    Input a dictionary of compared results from normal approx and bootstrap. You obtain this dictionary using the function "compare_methods" in
    the class ci_and_bootstrap.
    """
    normal = comparison_results["normal"]
    bootstrap = comparison_results["bootstrap"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = [0, 1]
    methods = ["Normal\napproximation", "Bootstrap"]
    
    # Normal approximation
    ax.errorbar([normal["mean"]], [0], 
                xerr=[[normal["mean"] - normal["lo"]], [normal["hi"] - normal["mean"]]], 
                fmt="o", markersize=8, capsize=10, capthick=2, 
                label="Normal", color="blue")
    
    # Bootstrap
    ax.errorbar([bootstrap["mean"]], [1], 
                xerr=[[bootstrap["mean"] - bootstrap["lo"]], [bootstrap["hi"] - bootstrap["mean"]]], 
                fmt="o", markersize=8, capsize=10, capthick=2, 
                label="Bootstrap", color="green")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods)
    ax.set_xlabel("Systolic Blood Pressure")
    ax.set_title("95% CI for mean of systolic_bp")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plotting_power_of_that_t_test(n_exp, n_control, alternative="greater"):
    """
    This plot visualizes how the statistical power of a t test changes depending on what difference in mmHg we are looking for.

    Input: Number of participants in experiment group, number of participants in control group, alternative="greater", "less" or "two-sided"
    """
    effect_sizes = np.linspace(0, 6, 25)  
    powers = []

    for diff in effect_sizes:
        power, _ = checking_power_of_that_t_test(
            n_experiment_group=n_exp,
            n_control_group=n_control,
            std_experiment_group=13.2678,
            std_control_group=12.626038,
            diff_to_find=diff,
            alpha=0.05,
            n_simulations=1000,
            alternative=alternative
        )
        powers.append(power)

    plt.figure(figsize=(10, 6))
    plt.plot(effect_sizes, powers, "b-", linewidth=2, label="Power curve")
    plt.axhline(y=0.8, color="r", linestyle="--", linewidth=1.5, label="80% power threshold")

    plt.xlabel("Difference to Find (mmHg)", fontsize=12)
    plt.ylabel("Statistical Power", fontsize=12)
    plt.title(f"Power Analysis: Effect Size vs Statistical Power\n(n_sick={n_exp}, n_healthy={n_control}, α=0.05)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim(0, 1)
    plt.xlim(0, 6)

    power_80_idx = np.argmin(np.abs(np.array(powers) - 0.8))
    effect_at_80 = effect_sizes[power_80_idx]
    
    if powers[power_80_idx] >= 0.75:
        plt.annotate(f"80% power at ~{effect_at_80:.1f} mmHg", 
                    xy=(effect_at_80, 0.8), 
                    xytext=(effect_at_80 + 0.5, 0.7),
                    arrowprops=dict(arrowstyle="->", color="red"),
                    fontsize=10)

    plt.tight_layout()
    plt.show()

    print(f"\n--- Summary ---")
    print(f"Effect size needed for 80% power: ~{effect_at_80:.2f} mmHg")