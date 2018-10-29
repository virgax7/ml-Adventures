import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn.linear_model as skl_lm
from scipy import stats
import statsmodels.formula.api as smf


### Simple Linear Regression

# read Advertising.csv
ads = pd.read_csv('Data/Advertising.csv', usecols=[1, 2, 3, 4])
print(ads.head(3))
ads_summary = ads.describe();
print(ads_summary)

# 3.1.1 Estimating the Coefficients

n = int(ads.size / 4)
tv_mean = ads_summary.TV[1]
sales_mean = ads_summary.Sales[1]
b1_numerator = 0
b1_denominator = 0
for i in range(n):
    b1_numerator += (ads.TV[i] - tv_mean) * (ads.Sales[i] - sales_mean)
    b1_denominator += (ads.TV[i] - tv_mean) ** 2

b1 = b1_numerator / b1_denominator
b0 = sales_mean - b1 * tv_mean

print("the simple linear fit for our advertising data using TV as our predictor takes the form: " + "y_sub_i = " + str(
    b0) + " + " + " " + str(b1) + "x_sub_i")

# Figure 3.1
sb.regplot(ads.TV, ads.Sales, ci=None, scatter_kws={'color': 'r', 's': 5})
plt.xlim(-7, 307)
plt.ylim(ymin=0);
plt.show()

# 3.1.2 Assessing the Accuracy of the Coefficient Estimates

# Equation 3.8
rss = 0
for i in range(n):
    rss += (ads.Sales[i] - (b0 + b1 * ads.TV[i])) ** 2
print("rss is ", rss)

rse = (rss / (n - 2)) ** (1/2)
print("rse is ", rse)

# rse is the σ
square_deviations_from_x_mean = 0
for i in range(n):
    square_deviations_from_x_mean += (ads.TV[i] - tv_mean) ** 2

# SE(βˆ1) ^2
squared_SE_of_B_sub_1 = rse ** 2 / square_deviations_from_x_mean

# 2 · SE(βˆ1).
two_SE_of_b1 = 2 * (squared_SE_of_B_sub_1 ** (1 / 2))

#βˆ1 ± 2 · SE(βˆ1).
b1_interval_range = "The 95% confidence interval range for B1 is [" + str(b1 - two_SE_of_b1) + "," + str(b1 + two_SE_of_b1) + "]"

# SE(βˆ0) ^2
squared_SE_of_B_sub_0 = rse ** 2 * (1 / n + tv_mean ** 2 / square_deviations_from_x_mean)

# 2 · SE(βˆ0).
two_SE_of_b0 = 2 * (squared_SE_of_B_sub_0 ** (1 / 2))
b0_interval_range = "The 95% confidence interval range for B0 is [" + str(b0 - two_SE_of_b0) + "," + str(b0 + two_SE_of_b0) + "]"
print(b1_interval_range)
print(b0_interval_range)

b0_t_statistic = b0 / (two_SE_of_b0 / 2)
b1_t_statistic =  b1 / (two_SE_of_b1 / 2)
b0_pval = stats.t.sf(np.abs(b0_t_statistic), n - 1) * 2
b1_pval = stats.t.sf(np.abs(b1_t_statistic), n - 1) * 2

print("               Coefficient, Std.Error, t-statistic, p-value")
print("Intercept ", b0, two_SE_of_b0 / 2, b0_t_statistic, b0_pval)
print("TV        ", b1, two_SE_of_b1 / 2, b1_t_statistic, b1_pval)

# 3.1.3 Assessing the Accuracy of the Model
# total sum of squares
tss = 0
for i in range(n):
    tss += (ads.Sales[i] - sales_mean) ** 2

r_squared = (tss - rss) / tss
print("r_squared is ", r_squared)

### Multiple Linear Regression

lr = skl_lm.LinearRegression()
tv_coef = ads.TV.values.reshape(-1,1)
radio_coef = ads.Radio.values.reshape(-1,1)
newspaper_coef = ads.Newspaper.values.reshape(-1,1)

# Table 3.3

print("Simple regression of sales on radio")
lr.fit(radio_coef, ads.Sales)
print("Intercept ", lr.intercept_)
print("radio", lr.coef_)

print("Simple regression of sales on newspaper")
lr.fit(newspaper_coef, ads.Sales)
print("Intercept ", lr.intercept_)
print("newspaper", lr.coef_)

# another way to do Table 3.3, as well as R-squared and etc
rlr = smf.ols("Sales ~ Radio", ads).fit()
print(rlr.summary())

# Table 3.4
mlr = smf.ols("Sales ~ TV + Radio + Newspaper", ads).fit()
print(mlr.summary())

# Table 3.5
print(ads.corr())

credits = pd.read_csv("Data/Credit.csv")

# sb.pairplot(credits[['Balance','Age','Cards','Education','Income','Limit','Rating']])
# plt.show()

# Table 3.7

# balance onto gender, but gender is originally a qualitive data(Male or Female)
glr = smf.ols("Balance ~ Gender", credits).fit()
print(glr.summary().tables[1])

# Table 3.8
elr = smf.ols("Balance ~ Ethnicity", credits).fit()
print(elr.summary().tables[1])


# Table 3.9
trlr = smf.ols("Sales ~ TV + Radio + Radio * TV", ads).fit()
print(trlr.summary().tables[1])


# Table 3.10
autos = pd.read_csv("Data/Auto.csv", na_values = "?").dropna()
autos["horsepower_squared"] = autos.horsepower ** 2
hh2lr = smf.ols("mpg ~ horsepower + horsepower_squared", autos).fit()
print(hh2lr.summary().tables[1])

# Table 3.10 without polynomial regression
hlr = smf.ols("mpg ~ horsepower", autos).fit()
print(hlr.summary().tables[1])













