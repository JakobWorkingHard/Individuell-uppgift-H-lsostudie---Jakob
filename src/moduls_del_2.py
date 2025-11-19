import matplotlib.pyplot as plt
from scipy import stats
from scipy import linalg
import pandas as pd
import math
from src.moduls import * 
from src.io_utils import *
from src.viz import *
import statsmodels.api as sm


# So is anything correlating with blood pressure?
# Let's take a look at Paul Allens... , i mean do a multiple regression with the independent variables:
# age, weight, height and cholesterol and check if any of these correlate with systolic_bp.
Y = df["systolic_bp"].values

X = df[["age", "height", "weight", "cholesterol"]].values

X_with_intercept = np.column_stack([np.ones(len(X)), X])

coefficients, residuals, rank, s = linalg.lstsq(X_with_intercept, Y)

print("Intercept ({beta_0}):", coefficients[0])
print("age coefficient ({beta_1}):", coefficients[1])
print("height coefficient ({beta_2}):", coefficients[2])
print("weight coefficient ({beta_3}):", coefficients[3])
print("cholesterol ({beta_4}):", coefficients[4])

y_pred = X_with_intercept @ coefficients

ss_res = np.sum((Y - y_pred) ** 2)
ss_tot = np.sum((Y - np.mean(Y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"\nR-squared: {r_squared:.4f}")

df["predicted_systolic_bp"] = y_pred
df["residuals"] = Y - y_pred

X_with_intercept = sm.add_constant(X)
model = sm.OLS(Y, X_with_intercept).fit()
print(model.summary())

#Ok so what the hell do these numbers mean? The regression coefficients tell us how much systolic BP changes for each unit increase in the independent variable, holding other variables constant.
#From the results, age and cholesterol have the largest coefficients (0.52 and 0.53), suggesting they might be the strongest predictors. However, these 4 variables in TOTAL explain about 40% of the variance in systolic BP, which means we have 60% unexplained variance.
#But when we look at the p-value, we see that x1 (age), and x3 (weight) are statistically significant. This means that, even though cholesterol had a large coefficient, it's not statistically significant, and therefore we have to drop it. So we are left with only age as a statistically significant variable with a high coefficient. Let's do the same multiple regression but using only age and weight, and see if R-squared drops or not. If it drops, there is something else going on. If it stays pretty much the same, then we can drop the other variables.


X_reduced = df[["age", "weight"]].values
X_reduced = sm.add_constant(X_reduced)

model_reduced = sm.OLS(Y, X_reduced).fit()
print(model_reduced.summary())

#Conclusion
#R-squared stayed the same, so we only focus on age and weight
#Next step: To see if age and weight have multicollinarity


from statsmodels.stats.outliers_influence import variance_inflation_factor

X_reduced = df[["age", "weight"]].values

vif_data = pd.DataFrame()
vif_data["Variable"] = ["age", "weight"]
vif_data["VIF"] = [variance_inflation_factor(X_reduced, i) for i in range(X_reduced.shape[1])]
print(vif_data)

#We have a VIF above 5, which is considered problematic. So age and weight are heavily influenced by the other.

