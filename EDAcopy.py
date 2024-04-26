#%%[markdown]
# NOTE: This is a copy of EDA First Steps, to use the variables created in this file for the modeling part.


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
df1 = pd.read_csv('DATS 6103 Final Team 2 Data.csv')

df1.head()
# Unnamed:, Case_ID, and region9 contain redundant index values, so they can be dropped
df1 = df1.drop(["Unnamed: 0", "case_ID"], axis = 1)
#%%
# Check proportions of missing values
col = df1.columns.to_list()

#%%[markdown]
# Seems that many are missing the exact same percentage of values
# %%
missing1 = df1['year'].isna()
df2 = df1[missing1]

# %%[markdown]
# Looks like they are, so we will move forward without those rows since they have no data in them
# %%
cleaning = df1[-missing1]
#%%[markdown]
# For marital status and employment, there were a higher number of missing values, so we'll check to see how many remain


# %%
missing2 = cleaning['marit_status'].isna()

df3 = cleaning[missing2]

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


# %%[markdown]
# It seems that 1 row of the income variable was incorrectly entered and only kept the last two digits instead of the full category (99 instead of $35,000 to $39,999 for example).  
# We will replace this value with the mode of income
# %%
incomecheck = df['income'] == "99"
df[incomecheck]['income']
# %%
df.iloc[30041, 5] = statistics.mode(df['income'])

#%%
# Checking outliers in odd looking distributions, such as spending
#%%
spendingcheck = df['spending'] > 5000
df[spendingcheck]['spending']
# It seems like there are genuinely several rows with high spending, and this variable is just very thinly distributed with a massive spike at 0. 
#%%[markdown]
# It appears that age is relatively normally distributed. Some of the variables are rather skewed, such as snowfall, adults, children, spending, disasters, and population. 
# QQplots for numeric variables
 
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
comp = 10
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
comp = 10
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
