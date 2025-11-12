import matplotlib.pyplot as plt
from src.moduls import *


def plot_gender_weight_difference(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8,5))
    df.boxplot(column="weight", by="sex", ax=ax)
    ax.set_title("Weight for males/females")
    ax.set_xlabel("")
    ax.set_ylabel("weight in kilos")
    plt.suptitle("")
    plt.tight_layout()
    plt.show


def plot_disease_vs_healthy(df1, df2: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10,8), sharey=False)

    df1.plot(kind="bar", ax=axes[0], legend=False)
    axes[0].set_title("Non disease-people")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Mean - Blood pressure")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].set_ylim(140,160)

    df2.plot(kind="bar", ax=axes[1], legend=False)
    axes[1].set_title("Disease-people")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean - Blood pressure")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].set_ylim(140,160)

    fig.suptitle("Comparing blood pressure levels of sick and healthy people with smoker/non smoker and sex")
    plt.tight_layout()
    plt.show()
    

def plot_comparing_age_with_cholesterol(df1, df2: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10,7))

    df1.plot(kind="scatter", ax=axes[0], legend=False, x="age", y="cholesterol", color="red" )
    axes[0].set_title("Comparing age with cholesterol - disease")
    axes[0].set_ylabel("Cholesterol")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].set_ylim(2,8)
    axes[0].set_xlim(0,90)

    df2.plot(kind="scatter", ax=axes[1], legend=False, x="age", y="cholesterol")
    axes[1].set_title("Comparing age with cholesterol - healthy")
    axes[1].set_ylabel("Cholesterol")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].set_ylim(2,8)
    axes[1].set_xlim(0,90)

    plt.tight_layout()
    plt.show()

def just_a_basic_plot(df: pd.DataFrame, x_label_and_category: str, y_label_and_category: str):
    fig, ax = plt.subplot(figsize=(10,8))
    df.bar(x_label_and_category, y_label_and_category, color="seagreen")
    ax.set_title("")
    ax.set_xlabel(x_label_and_category)
    ax.set_ylabel(y_label_and_category)
    plt.tight_layout()
    plt.show()

def ci_mean_and_ci_bootstrap(df: pd.DataFrame):
    lo, hi, x_mean, s, n = ci_mean_norma(df)
    blo, bhi, b_mean = ci_mean_bootstrap(df)
    bm = np.array([np.mean(np.random.choice(df, size=len(df), replace=True)) for _ in range(1000)])
    fig, axes = plt.subplots(1, 2, figsize=(10,8))

    axes[0].errorbar([0], [x_mean], yerr=[[x_mean - lo], [hi - x_mean]], fmt="o", capsize=6)
    axes[0].errorbar([1], [b_mean], yerr=[[b_mean - blo], [bhi - b_mean]], fmt="o", capsize=6, color="r", label="Bootstrap CI")
    axes[0].set_xticks([])
    axes[0].grid(True, axis="y")
    axes[0].legend()
    axes[0].set_title("95% konfidensintervall med normalapproximation")

    axes[1].hist(bm, bins=30, edgecolor="black")
    axes[1].axvline(x_mean, color="tab:green", linestyle="--", label="Stickprovsmedel")
    axes[1].axvline(np.percentile(bm, 2.5), color="tab:red", linestyle="--", label="2,5%")
    axes[1].axvline(np.percentile(bm, 97.5), color="tab:red", linestyle="--", label="97,5%")
    axes[1].set_title("Bootstrap fördelninga av medel + 95 procent -intervall")
    axes[1].set_xlabel("Resamplat medel")
    axes[1].set_ylabel("Antal")
    axes[1].grid(True, axis="y")

