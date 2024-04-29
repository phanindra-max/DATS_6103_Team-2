#%%[markdown]
# Below are some first steps of EDA, which includes:
# * Summary Statistics of the Variables in our dataset
# * Data Visualization of the distribution of the Variables in our dataset
# * Outlier and Missing Value Detection, and Imputation
# * Feature Engineering (converting categorical variables into dummy coded versions)
# * PCA for dimensionality reduction
# * Statistical Tests for categorical variables - T-tests for binary variables, ANOVA for 3+ categories
#%%[markdown]
# Dataframes that are created in this file and ready for next steps include:
# * "df"
# * "df_withdummies"
# * "df_PCA"
# * PCA_comps1 contains the coefficients for the PCA from the slides, while df_PCA contains the values for the principal components. Currently, the dataframe does not have the categorical variables, so those would need to be combined to make a larger dataframe
# * "dummy_PCA"

#%%
import pandas as pd
import numpy as np
import statistics
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import pylab as py
# %%
# Import data
df1 = pd.read_csv('../DATS 6103 Final Team 2 Data.csv')

df1.head()
# Unnamed:, Case_ID, and region9 contain redundant index values, so they can be dropped
df1 = df1.drop(["Unnamed: 0", "case_ID"], axis = 1)
#%%
# Check proportions of missing values
col = df1.columns.to_list()

for cols in col:
  print(f"Percentage of missing values in {cols}: {100 * df1[cols].isna().sum() / len(df1)}")
#%%[markdown]
# Seems that many are missing the exact same percentage of values
# %%
missing1 = df1['year'].isna()
df2 = df1[missing1]
#%%
# Checking if those missing values are missing in all columns
for cols in col:
  print(f"Percentage of missing values in {cols}: {100 * df2[cols].isna().sum() / len(df2)}")
# %%[markdown]
# Looks like they are, so we will move forward without those rows since they have no data in them
# %%
cleaning = df1[-missing1]
#%%[markdown]
# For marital status and employment, there were a higher number of missing values, so we'll check to see how many remain
# %%
for cols in col:
  print(f"Percentage of missing values in {cols}: {100 * cleaning[cols].isna().sum() / len(cleaning)}")
# %%
for cols in col:
  print(f"Summary Statistics for {cols}: {cleaning[cols].describe()}")


# %%
missing2 = cleaning['marit_status'].isna()

df3 = cleaning[missing2]

#%%
for cols in col:
  print(f"Summary for {cols}: {df3[cols].describe()}")
# %%[markdown]
# After comparing the summary statistics of the missing values in marital and employment to the entire dataset, it seems that they stopped collecting marital and employment data for 2021 and onward, when analysing marital and employment data, our analysis will therefore be limited to those years.
#%%
# Our final dataset will be a copy of the cleaned dataset
df = cleaning.copy(deep = True)

#%%
# Lastly, the "Wave" column gives us the survey wave that the information is given in, telling us the month and the year, but sine 'Year' already gives us the year of the survey, we will convert 'Wave' into 'Month' 
df['month'] = df['wave'].str[0:3]
df = df.drop(['wave', 'region9'], axis = 1)

# %%
# Now that missing values have been treated, we will dive more in depth to the summary statistics of each variable
# Beginning with numeric variables:
num_cols = ['age', 'c_temp', 'snowfall', 'rainfall', 'disasters', 'storms', 'spending', 'el_nino', 'g_temp', 'g_temp_lowess', 'children', 'adults', 'population']
# Case_ID is not on this list, as it seems to be another index column that does not provide any useful information
cat_cols = ['happening', 'female', 'education', 'income', 'race', 'ideology', 'party', 'religion', 'marit_status', 'employment', 'City', 'year', 'month']


#%%
for cols in num_cols:
  print(f"Summary Statistics for {cols}: {df[cols].describe()}")
  print(f"")
# %%
for cols in cat_cols:
  print(f"Summary Statistics for {cols}")
  print(f"Mode: {statistics.mode(df[cols])}")
  print(f"Frequency of each unique category: {df[cols].value_counts()}")
  print(f"")
# %%[markdown]
# It seems that 1 row of the income variable was incorrectly entered and only kept the last two digits instead of the full category (99 instead of $35,000 to $39,999 for example).  
# We will replace this value with the mode of income
# %%
incomecheck = df['income'] == "99"
df[incomecheck]['income']
# %%
df.iloc[30041, 5] = statistics.mode(df['income'])
#%%
# Visualizing distributions
for cols in num_cols:
    sns.histplot(data= df, x = cols, bins = 15)
    plt.show()

#%%
# Checking outliers in odd looking distributions, such as spending
#%%
spendingcheck = df['spending'] > 5000
df[spendingcheck]['spending']
# It seems like there are genuinely several rows with high spending, and this variable is just very thinly distributed with a massive spike at 0. 
#%%[markdown]
# It appears that age is relatively normally distributed. Some of the variables are rather skewed, such as snowfall, adults, children, spending, disasters, and population. 
# QQplots for numeric variables
# %%
for cols in num_cols:
  sm.qqplot(df[cols], line= 's')
  py.title(f"{cols}") 
  py.show()   
# %%[markdown]
# As you can see, many of the numeric variables are heavily skewed, and not normally distributed
#%%
# Visualizing Categorical Data
# %%
df['income'] = df['income'].astype("category")
# %%
cats_income = ["Less than $5,000", 
               "$5,000 to $7,499",
               "$7,500 to $9,999",
               "$10,000 to $12,499",
               "$12,500 to $14,999",
               "$15,000 to $19,999",
               "$20,000 to $24,999",
               "$25,000 to $29,999",
               "$30,000 to $34,999",
               "$35,000 to $39,999",
               "$40,000 to $49,999",
               "$50,000 to $59,999",
               "$60,000 to $74,999",
               "$75,000 to $84,999",
               "$85,000 to $99,999",
               "$100,000 to $124,999",
               "$125,000 to $149,999",
               "$150,000 to $174,999",
               "$175,000 to $199,999 (Nov 2016 on); $175,000 or more (Nov 2008 - Mar 2016)",
               "$200,000 to $249,999 (Nov 2016 on)",
               "$250,000 or more (Nov 2016 on)"]
order_income = pd.CategoricalDtype(cats_income, ordered=True)
df['income']=df['income'].astype(order_income)

#%%
cats_education = ["Bachelor's degree or higher",
                  "Some college",
                  "High school",
                  "Less than high school"]
order_education = pd.CategoricalDtype(cats_education, ordered=True)
df['education'] = df['education'].astype(order_education)
#%%
cats_employ = ["Working - as a paid employee",
               "Working - self-employed",
               "Not working - on temporary layoff from a job",
               "Not working - looking for work",
               "Not working - disabled",
               "Not working - retired",
               "Not working - other"]
order_employ = pd.CategoricalDtype(cats_employ, ordered=True)
df['employment'] = df['employment'].astype(order_employ)
#%%
cats_month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
order_month = pd.CategoricalDtype(cats_month, ordered=True)
df['month'] = df['month'].astype(order_month)
#%%
df['year'] = df['year'].astype("int")
df['female'] = df['female'].astype("int")
df['happening'] = df['happening'].astype("int")
# %%
for cols in cat_cols:
  sns.countplot(data= df, y = cols)
  plt.show()
# %%[markdown]
# Our categorical variables have a variety of distributions, most of which are not evenly spread between categories. Gender, however, is a roughly 50/50 split, and the year variable has similar frequencies from 2008 to 2021, but drops off in 2022. All of our other variables, however, have significant amounts of variation in the frequency of each category. Notably, our outcome variable, opinion on whether climate change is or is not happening, is a roughly 67/33 split, with roughly two thirds of all respondents from between 2008 to 2023 believing in climate change.

# %%
# Feature Engineering for model building
# Creating dummy variables for categorical variables
dummies = []
dummy_cat_cols = cat_cols.copy()
dummy_cat_cols.remove('female')
dummy_cat_cols.remove('happening')
dummy_cat_cols.remove('year')
#%%
for cols in dummy_cat_cols:
  cols_dummy = pd.get_dummies(df[cols],
                                drop_first=True,
                                dtype = int)
  dummies.append(cols_dummy)

# len(dummies)
# %%
df_withdummies = pd.concat([df.drop(dummy_cat_cols, axis = 1), dummies[0], dummies[1], dummies[2], dummies[3], dummies[4], dummies[5], dummies[6], dummies[7], dummies[8], dummies[9]], axis = 1)
#%%
# Creating Principal Components for PCA
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
con_df = df[num_cols]
cat_df = df[cat_cols]
cat_dummy_df = df_withdummies.drop(num_cols, axis=1)
#%%
con_df_s = sc.fit_transform(con_df)
# %%
con_df_s = pd.DataFrame(con_df_s)
con_df_s = con_df_s.rename(columns={0:"age", 
                         1:"c_temp",
                         2:"snowfall",
                         3:"rainfall",
                         4:"disasters",
                         5:"storms",
                         6:"spending",
                         7:"el_nino",
                         8:"g_temp",
                         9:"g_temp_lowess",
                         10:"children",
                         11:"adults",
                         12:"population"})
# %%
con_df_s.index += 1
#%%
df_standard = pd.concat([con_df_s, cat_df], axis = 1)
dummy_standard = pd.concat([con_df_s, cat_dummy_df], axis = 1)
# %%
df_standard.head()
dummy_standard.head()
# %%
# Number of components chosen to provide 95% explained variance for the first analysis
from sklearn.decomposition import PCA
comp = 5
pca = PCA(n_components= comp)
#%%
df_PCA = pca.fit_transform(con_df_s)
df_PCA = pd.DataFrame(df_PCA)
pca.explained_variance_ratio_
#%%
sum(pca.explained_variance_ratio_)
#%%
PCA_comps1 = []
PCA_comps1 =+ pca.components_
PCA_comps1
#%%
comp = 5
pca = PCA(n_components= comp)
dummy_PCA = pca.fit_transform(dummy_standard)
dummy_PCA = pd.DataFrame(dummy_PCA)
sum(pca.explained_variance_ratio_)
# %%
PCA_comps2 = []
PCA_comps2 =+ pca.components_
PCA_comps2[0]
# %%[markdown]
# Our PCA of the original dataset, before converting categorical variables to dummy variables, shows that the first principal component associates storms, el_nino, g_temp, and g_temp_lowess with one another, and going the other direction is snowfall in the first principal component. Our second principal component shows c_temp and snowfall move in opposite directions from one another. Our third component shows that disasters and spending are paired together. Our fourth component shows that age and children go in opposite directions. The remaining components explain relatively low amounts of variance individually, so it is difficult to draw meaningful relationships as described above (for example, PC5 shows a diverging relationship between rainfall and populations, but PC7 shows the opposite).
# %%[markdown]
# In an attempt to make the output interpretable, the explained variance for the PCA of the dataset with dummy coded variables was only required to be 80%, giving us once again 10 components. However, all of the coefficients are nearly 0, and each component has 106 coefficients, which makes them nearly impossible to meaningfully interpret. Since this does not provide much information, the PCA will switch to a larger amount of explained variance by incorporating more components, so that a reduced dimensionality dataset that explains a large amount of variance can be created
# %%
comp = 40
pca = PCA(n_components= comp)
dummy_PCA2 = pca.fit_transform(dummy_standard)
dummy_PCA2 = pd.DataFrame(dummy_PCA2)
sum(pca.explained_variance_ratio_)
# %%[markdown]
# Dataframes ready to use for modeling
# df
# df_withdummies
# df_PCA
# dummy_PCA2
# potentially dummy_PCA if dummy_PCA2 is too annoying to work with
# %%
# Function to run t-tests
def t_test2(data, cat, cont):
  sample1_bool = data[cat] == 0
  sample1 = data[sample1_bool][cont]
  sample2_bool = data[cat] == 1
  sample2 = data[sample2_bool][cont]
  stat, p = stats.ttest_ind(sample1, sample2)
  print(f"Stat: {stat:.2f}")
  print(f"p-value: {p:.2f}")
  return
# %%
# t_test2('happening', 'age')
# %%
# Function to run anova
def anova(data, cat, cont):
  groups = list(data[cat].unique())
  var = []
  for group in groups:
    bool = data[cat] == group
    slices = data[bool][cont]
    var.append(slices)
  stat, p = stats.f_oneway(*var)
  print(f"Stat: {stat:.2f}")
  print(f"p-value: {p:.2f}")
  return
#%%
# anova(df, 'income', 'age')
#%%
# Because of missing values in Marital Status and Employment, they need to be treated differently
# Marital Status
em_bool = df['year'] < 2021
df_em = df[em_bool]
#%%
#%%
# T-tests
# Happening
for nums in num_cols:
  print(nums)
  t_test2(df, 'happening', nums)
  print()
# %%
# Female
for nums in num_cols:
  print(nums)
  t_test2(df, 'female', nums)
  print()
# %%
# ANOVA
# Education
for nums in num_cols:
  print(nums)
  anova(df, 'education', nums)
  print()
#%%
# Income
for nums in num_cols:
  print(nums)
  anova(df, 'income', nums)
  print()
#%%
# Race
for nums in num_cols:
  print(nums)
  anova(df, 'race', nums)
  print()
#%%
# Ideology
for nums in num_cols:
  print(nums)
  anova(df, 'ideology', nums)
  print()
#%%
# Party
for nums in num_cols:
  print(nums)
  anova(df, 'party', nums)
  print()
#%%
# Religion
for nums in num_cols:
  print(nums)
  anova(df, 'religion', nums)
  print()
#%%
# Marital Status
for nums in num_cols:
  print(nums)
  anova(df_em, 'marit_status', nums)
  print()
#%%
# Employment
for nums in num_cols:
  print(nums)
  anova(df_em, 'employment', nums)
  print()

#%%
# City
for nums in num_cols:
  print(nums)
  anova(df, 'City', nums)
  print()
#%%
# Year
for nums in num_cols:
  print(nums)
  anova(df, 'year', nums)
  print()
#%%
# Month
for nums in num_cols:
  print(nums)
  anova(df, 'month', nums)
  print()
# %%[markdown]
# Investigating the results of our tests
# It seems that the vast majority of all tests were significant, so I want to see if there are any structural properites of our data that would lead to that

#%%[markdown]
# It seems that global temp varies by city, which doesn't make sense, so let's look at these variables
#%%
sns.violinplot(df, x = 'g_temp', y= 'City')
#%%[markdown]
# Perhaps cities are being surveyed in different years, which could lead to some sort of effect like this
sns.countplot(df, y = 'City', hue = 'year')

#%%[markdown]
# Based on the above visualizations, it seems that perhaps the differences we are detecting do exist, but perhaps they are simply small, but are relayed as significant because of our large sample size
#%%
# Measuring effect size
from statsmodels.stats.oneway import effectsize_oneway
#%%
def anova(data, cat, cont):
  groups = list(data[cat].unique())
  var = []
  for group in groups:
    bool = data[cat] == group
    slices = data[bool][cont]
    var.append(slices)
  stat, p = stats.f_oneway(*var)
  means = []
  variance = []
  for vars in var:
    m = statistics.mean(vars)
    means.append(m)
    v = statistics.variance(vars)
    variance.append(v)
  e = effectsize_oneway(means, variance, nobs= len(data), use_var="equal") 
  print(f"Stat: {stat:.2f}")
  print(f"p-value: {p:.2f}")
  print(f"Effect size: {e:.2f}")
  return
#%%
# Rerunning ANOVA to include effect sizes
# Education
for nums in num_cols:
  print(nums)
  anova(df, 'education', nums)
  print()
#%%
# Income
for nums in num_cols:
  print(nums)
  anova(df, 'income', nums)
  print()
#%%
num_cols
#%%
# Race
for nums in num_cols:
  print(nums)
  anova(df, 'race', nums)
  print()
#%%
# Ideology
for nums in num_cols:
  print(nums)
  anova(df, 'ideology', nums)
  print()
#%%
# Party
for nums in num_cols:
  print(nums)
  anova(df, 'party', nums)
  print()
#%%
# Religion
for nums in num_cols:
  print(nums)
  anova(df, 'religion', nums)
  print()
#%%
# Marital Status
for nums in num_cols:
  print(nums)
  anova(df_em, 'marit_status', nums)
  print()
#%%
# Employment
for nums in num_cols:
  print(nums)
  anova(df_em, 'employment', nums)
  print()

#%%
# City
for nums in num_cols:
  print(nums)
  anova(df, 'City', nums)
  print()
#%%
# Year
for nums in num_cols:
  print(nums)
  anova(df, 'year', nums)
  print()
#%%
# Month
for nums in num_cols:
  print(nums)
  anova(df, 'month', nums)
  print()
# %%
# Create function to calculate t-test effect size
from numpy import std, mean, sqrt
def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)

# %%
# Restructuring t-test to incorporate effect size
def t_test2(data, cat, cont):
  sample1_bool = data[cat] == 0
  sample1 = data[sample1_bool][cont]
  sample2_bool = data[cat] == 1
  sample2 = data[sample2_bool][cont]
  stat, p = stats.ttest_ind(sample1, sample2)
  e = cohen_d(sample1, sample2)
  print(f"Stat: {stat:.2f}")
  print(f"p-value: {p:.2f}")
  print(f"Effect size: {e:.2f}")
  return
# %%
# Happening
for nums in num_cols:
  print(nums)
  t_test2(df, 'happening', nums)
  print()
# %%
# Female
for nums in num_cols:
  print(nums)
  t_test2(df, 'female', nums)
  print()

#%%
#df_PCA.head()
# %%
#PCA_comps1
# %%
# Aligning PCA dataframe with overall dataframe
df_PCA += 1
#%%
# Combining PCA dataframe with categorical variables
df_pca_cat = pd.concat([df_PCA, cat_df], axis = 1)
df_pca_catasdummies = pd.concat([df_PCA, cat_dummy_df], axis=1)
#%%
