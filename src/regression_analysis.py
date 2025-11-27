import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


class RegressionAnalysis:
    def __init__(self, data: pd.DataFrame):
        """
        Initiates the self using a dataframe as self.data
        """
        self.data = data.copy()

    def linear_regression(self, independent_variable: str, dependent_variable: str):
        """
        Doing a linear regression using independent variable and dependent variable
        Returns a tuple with slope, intercept, r2
        """
        Y = self.data[dependent_variable].values
        X = self.data[[independent_variable]].values
        X = sm.add_constant(X)
        
        model = sm.OLS(Y, X).fit()
        
        intercept = float(model.params[0])
        slope = float(model.params[1])
        r2 = float(model.rsquared)
        
        print(f"""
    Linear regression:
        Intercept = {intercept:.2f}
        Slope = {slope:.2f}
        R² = {r2:.3f}
        """)
        
        return slope, intercept, r2, model
    
    def multiple_regression(self, dependent_variable: str, independent_variable_list: list):
        """
        Doing a multiple regression using independent variables in a list and a dependent variable as a string.  
        Note that this function only works if all the variables are in the dataframe used in the class self.
        Returns a dictionary of coefficients, r2, predictions and residuals
        """
        Y = self.data[dependent_variable].values
        X = self.data[independent_variable_list].values
        X = sm.add_constant(X)
        
        model = sm.OLS(Y, X).fit()
        
        print(f"Intercept (β₀): {model.params[0]:.4f}")
        for i, var_name in enumerate(independent_variable_list):
            print(f"{var_name} (β{i+1}): {model.params[i+1]:.4f}")
        
        print(f"\nR-squared: {model.rsquared:.4f}")
        
        print("\n" + "="*60)
        print(model.summary())
        
        return {
            'coefficients': model.params,
            'r2': model.rsquared,
            'predictions': model.fittedvalues,
            'residuals': model.resid
        }
    
    def calculate_vif(self, independent_variable_list: list):
        """
        Calculating VIF (Variance Inflation Factor) to detect multicollinearity among your choice of independent variables.
        Returns a dataframe of the VIF-values of each variable
        """
        X = self.data[independent_variable_list].values
        
        vif_data = pd.DataFrame()
        vif_data["Variable"] = independent_variable_list
        vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

        print("--> VARIANCE INFLATION FACTOR (VIF) <-- Woohooo")
        print(vif_data)
        print("\nInterpretation:")
        print("VIF < 2: Small to none multicollinearity")
        print("VIF 5-10: Kind of high and needs to be investigated")
        print("VIF > 10: Way too high and now we're beginning to panic")
        
        return vif_data
    