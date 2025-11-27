from src.regression_analysis import *

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression 

import matplotlib.pyplot as plt
from scipy import stats
from scipy import linalg
import math
from src.moduls import * 
from src.io_utils import *
from src.viz import *
from src.viz_part2 import *
import statsmodels.api as sm

class PCA_and_PCR:
    """
    This class is capable of doing PCA (Principal component analysis) and PCR (Principal component regression).
    """
    def __init__(self, pca_vars: list, df: pd.DataFrame):
        """
        Initiates a init with pca_vars: a list of variables we want to do a PCA on. You also need to input your dataframe as it is.
        """
        self.pca_vars = pca_vars
        self.df = df

        X = self.df[self.pca_vars]
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        
        self.pca_model = None
        self.X_pca = None


    def pca(self, n_components=None):
        """
        This function executes a PCA (Principal component analysis) using the classes pca_vars (your list of variables) and your dataframe.

        Input: If you want, you can specify how many PCA components you want, otherwise it will return 4 PCA components.

        Returns: a dataframe of the PCA components and the variables they consist of, together with a coefficient of how largely the variables
        inpact the PCA-component (what constitutes each PCA component).
        """
        self.pca_model = PCA(n_components=n_components)
        self.X_pca = self.pca_model.fit_transform(self.X_scaled)
        explained_variance_ratio = self.pca_model.explained_variance_ratio_

        print("Explained variance per component:")
        for i, ratio in enumerate(explained_variance_ratio):
            print(f"PC{i+1}: {ratio:.3f} ({ratio*100:.1f}%)")
        
        cumulative_variance = explained_variance_ratio.cumsum()
        if len(cumulative_variance) >= 2:
            print(f"\nCumulative variance (PC1 + PC2): {cumulative_variance[1]:.3f} ({cumulative_variance[1]*100:.1f}%)")

        n_components_actual = self.pca_model.n_components_
        loadings_df = pd.DataFrame(
            self.pca_model.components_.T,                 
            columns=[f'PC{i+1}' for i in range(n_components_actual)],
            index=self.pca_vars
        )
        print("\nComponents:")
        print(loadings_df.round(3))
        return loadings_df


    def pcr(self, dependent_variable: str, n_components_to_use: int = 3):
        """
        This function executes a PCR (Principal component regression), a regression using your new PCA components.

        Input: Choose your dependent variable and number of PCA components you want to use in your regression.

        Returns: Nothing, only printed information about your regression.
        """
        X_pca_subset = self.X_pca[:, :n_components_to_use]
        X_pca_df = pd.DataFrame(
            data=X_pca_subset, 
            columns=[f'PC{i+1}' for i in range(n_components_to_use)],
            index=self.df.index  
        )
        y_target = self.df[dependent_variable]
    
        pcr_model = LinearRegression()
        pcr_model.fit(X_pca_df, y_target)

        print("--Principal Component Regression (PCR) Result--")
        print(f"Dependent variable (Y): {dependent_variable}")
        print(f"Independent variables (X): {', '.join([f'PC{i+1}' for i in range(n_components_to_use)])}")
        print("-" * 50)
        print(f"Intercept (β₀): {pcr_model.intercept_:.4f}")
        print(f"Coefficients (β):")
        for i, beta in enumerate(pcr_model.coef_):
            print(f"  PC{i+1}: {beta:.4f}")
        
        explained_variance = self.pca_model.explained_variance_ratio_[:n_components_to_use].sum()
        print("-" * 50)
        print(f"Variance explained of {n_components_to_use} components: {explained_variance:.4f} ({explained_variance*100:.1f}%)")
        
        r_squared_pcr = pcr_model.score(X_pca_df, y_target)
        print(f"R-squared for this beautiful PC-modell: {r_squared_pcr:.4f}")

        