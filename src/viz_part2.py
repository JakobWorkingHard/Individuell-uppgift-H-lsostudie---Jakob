
import matplotlib.pyplot as plt
from src.moduls import *
import numpy as np
from scipy import stats
import statsmodels.api as sm

def plotting_that_linear_regression(df: pd.DataFrame, x_axis: str, y_axis: str, slope: float = None, intercept: float = None):
    """
    This function plots a linear function within a scatter plot using inputs of a dataframe, x and y-variables, and 
    the linear functions information slope and intercept. The linear functions label is hardcoded"linear regression"
    which makes this function only suitable for plotting a linear regression line.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    df.plot(kind="scatter", ax=ax, legend=False, x=x_axis, y=y_axis, color="blue", alpha=0.6)

    if slope is not None and intercept is not None:
            x_min = df[x_axis].min()
            x_max = df[x_axis].max()
            
            grid_x = np.linspace(x_min, x_max, 100)
            grid_y_reg = intercept + slope * grid_x
    
            ax.plot(
                grid_x, 
                grid_y_reg, 
                color="red",        
                linewidth=2, 
                label="Linear regression (OLS)"
            )

    ax.set_title(f"Comparing {x_axis} with {y_axis}")
    ax.set_ylabel(y_axis)
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout
    plt.show()


def plot_residual_diagnostics(model, title="Residual Diagnostics"):
    """
    This function plots 4 different residual diagnostics. The information needed is the linear regressions model. 
    Above each residual diagnostics plot is a title and a short explanation of what the plot shows.
    """
    
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Residuals vs Fitted (checking linearity & homoscedasticity)
    axes[0, 0].scatter(fitted_values, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    # 2. Q-Q plot (checking normality / checking normal distribution)
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot')
    
    # 3. Scale-Location (checking homoscedasticity)
    influence = model.get_influence() 
    standardized_resid = influence.resid_studentized_internal 
    axes[1, 0].scatter(fitted_values, standardized_resid, alpha=0.5) 
    axes[1, 0].set_xlabel('Fitted values')
    axes[1, 0].set_ylabel('√|Standardized Residuals|')
    axes[1, 0].set_title('Scale-Location')
    
    # 4. Histogram of residuals (checking normality / checking normal distribution)
    axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Histogram of Residuals')
    
    plt.suptitle(title, fontsize=14, y=1.00)
    plt.tight_layout()
    plt.show()