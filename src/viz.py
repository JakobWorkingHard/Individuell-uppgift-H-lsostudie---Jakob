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
    axes[0].set_ylabel("Blood pressure")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].set_ylim(140,160)

    df2.plot(kind="bar", ax=axes[1], legend=False)
    axes[1].set_title("Disease-people")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Blood pressure")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].set_ylim(140,160)

    fig.suptitle("Comparing blood pressure levels of sick and healthy people with smoker/non smoker and sex")
    plt.tight_layout()
    plt.show()
    