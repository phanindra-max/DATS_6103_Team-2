#%%[markdown]
# Below are some first steps of EDA, which includes:
# * Correlation Analysis to check if and how are the variables correlated
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
import statistics
import numpy as np
import pylab as py
import pandas as pd
import seaborn as sns
import geopandas as gpd
import scipy.stats as stats
import statsmodels.api as sm 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# %%
# Import data
df1 = pd.read_csv('DATS 6103 Final Team 2 Data.csv')

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

#%%
#Correlation Analysis
num_cols_new = ['age', 'c_temp', 'snowfall', 'rainfall', 'disasters', 'storms', 'spending', 'el_nino', 'g_temp', 'children', 'adults', 'population']

#Scaling the numeric variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[num_cols_new])

# Calculating the correlation matrix
correlation_matrix = df[num_cols_new].corr()

# Displaying the correlation matrix
print(correlation_matrix)

plt.figure(figsize=(12, 8))

# %%[markdown]
#Insights from correlation Analysis
# * Age and Climate Temperature (c_temp): There is a very weak negative correlation between age and climate temperature.
# * Climate Temperature (c_temp) and Snowfall: There is a strong negative correlation between climate temperature and snowfall, which is expected as colder temperatures are associated with more snowfall.
# * Climate Temperature (c_temp) and Rainfall: There is a weak negative correlation between climate temperature and rainfall.
# * Climate Temperature (c_temp) and Natural Disasters (disasters): There is a weak positive correlation between climate temperature and the occurrence of natural disasters.
# * Storms and Natural Disasters (disasters): There is a weak negative correlation between the number of storms and the occurrence of natural disasters.
# * Storms and Climate Temperature (c_temp): There is a moderate positive correlation between the number of storms and climate temperature, indicating that higher temperatures may lead to more severe storms.
# * Spending and Natural Disasters (disasters): There is a moderate positive correlation between spending and the occurrence of natural disasters, suggesting that more spending occurs in regions with more disasters.
# * El Niño and Storms: There is a moderate positive correlation between the El Niño index and the number of storms, indicating that El Niño conditions may lead to more severe storms.
# * Global Temperature (g_temp) and Storms: There is a strong positive correlation between global temperature and the number of storms, suggesting that global temperature may influence storm frequency.
# * Global Temperature (g_temp) and Climate Temperature (c_temp): There is a strong positive correlation between global temperature and climate temperature, which is expected as global temperature affects local climate.


# Create a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".3f")

# Add title and display the plot
plt.title('Correlation Matrix')
plt.show()

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
con_df = df[num_cols]
cat_df = df[cat_cols]
cat_dummy_df = df_withdummies.drop(num_cols, axis=1)
#%%
sc = StandardScaler()
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
# # Part 2 of EDA


#%%[markdown]
# Below are some next steps of EDA, which includes:
# * Temporal Analysis to check any treds over years
# * Data Imbalance check for the Target variable
# * Data Standardization for modeling
# * Geographical Analysis to understand geographical patterns



# %%
# Import data
print(df1.head())

# %%
#Temporal Analysis
new_df = df1.loc[:,['year','age','c_temp','snowfall','rainfall','disasters','storms','spending']]

# Aggregating data based on year
yearly_data = new_df.groupby('year').mean()

# # Plotting the trends over time
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 15))

plots = [
    ('age', 'Average Age Over Time', 'Age'),
    ('c_temp', 'Average Temperature Over Time', 'Temperature'),
    ('snowfall', 'Average Snowfall Over Time', 'Snowfall'),
    ('rainfall', 'Average Rainfall Over Time', 'Rainfall'),
    ('disasters', 'Average Disasters Over Time', 'Disasters'),
    ('storms', 'Average Storms Over Time', 'Storms')
]

for i, (col, title, ylabel) in enumerate(plots):
    row, col_idx = i // 2, i % 2
    axes[row, col_idx].plot(yearly_data.index, yearly_data[col], marker='o')
    axes[row, col_idx].set_title(title)
    axes[row, col_idx].set_xlabel('Year')
    axes[row, col_idx].set_ylabel(ylabel)

plt.tight_layout()
plt.show()

# %%[markdown]

#Based on the temporal analysis of the data, the following conclusions can be drawn:
# * Age: The average age of the population seems to be increasing steadily over the years after a sharp fall in the year 2010. 
# * Average Temperature (c_temp): There is a clear upward trend in average temperatures over time, indicating a potential long-term increase in temperature.
# * Snowfall: The average snowfall shows a decreasing trend over the years, suggesting a possible decrease in snowfall amounts over time.
# * Rainfall: The average rainfall appears to decrease initially and then stabilize at a lower level witn increase in the year 2017, indicating a potential change in rainfall patterns over time.
# * Natural Disasters: The average number of natural disasters shows a steady trend with a a steep rise in the 2020, suggesting a potential increase in the frequency of these events over time.
# * Storms: The average number of severe storms shows an increasing trend, indicating a potential increase in the frequency of these events over time.

#%%
# Group by 'region9' and calculate mean for 'rainfall' and 'snowfall'
geographical_analysis_region = df1.groupby(['region9'])[['rainfall', 'snowfall', 'c_temp', 'disasters', 'storms']].mean().reset_index()

# Group by 'city' and calculate mean for 'rainfall' and 'snowfall'
geographical_analysis_city = df1.groupby(['City'])[['rainfall', 'snowfall', 'c_temp', 'disasters', 'storms']].mean().reset_index()

# Function to create a bar plot for region-based analysis
def create_region_bar_plot(y, title, ylabel):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='region9', y=y, data=geographical_analysis_region)
    plt.title(title)
    plt.xlabel('Region')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to create a bar plot for city-based analysis
def create_city_bar_plot(y, title, ylabel):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='City', y=y, data=geographical_analysis_city)
    plt.title(title)
    plt.xlabel('City')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Create bar plots for each variable by region9
create_region_bar_plot('rainfall', 'Mean Rainfall by Region', 'Mean Rainfall')
create_region_bar_plot('snowfall', 'Mean Snowfall by Region', 'Mean Snowfall')
create_region_bar_plot('c_temp', 'Mean Temperature by Region', 'Mean Temperature')
create_region_bar_plot('disasters', 'Mean Disasters by Region', 'Mean Disasters')
create_region_bar_plot('storms', 'Mean Severe Storm Warnings by Region', 'Mean Storms')

# Create bar plots for each variable by city
create_city_bar_plot('rainfall', 'Mean Rainfall by City', 'Mean Rainfall')
create_city_bar_plot('snowfall', 'Mean Snowfall by City', 'Mean Snowfall')
create_city_bar_plot('c_temp', 'Mean Temperature by City', 'Mean Temperature')
create_city_bar_plot('disasters', 'Mean Disasters by City', 'Mean Disasters')
create_city_bar_plot('storms', 'Mean Severe Storm Warnings by City', 'Mean Storms')

# %%[markdown]
# Inference basis geospatial analysis
# * Temperature Variability: Cities and regions exhibit varying average temperatures, with some regions experiencing consistently higher or lower temperatures compared to others. This variability could be due to geographical location, elevation, or other local climate factors.
# * Snowfall Patterns: Similarly, there are differences in average snowfall across regions and cities, with some areas experiencing heavier snowfall than others. This could be influenced by proximity to bodies of water, latitude, or elevation.
# * Rainfall Distribution: Rainfall patterns also vary across regions and cities, with some areas receiving higher average rainfall than others. This could be influenced by proximity to oceans or other large bodies of water, as well as local topography.
# * Natural Disaster Frequency: The frequency of weather-related natural disasters varies across regions and cities, with some areas experiencing more frequent disasters than others. This could be due to factors such as geographical location, climate, and susceptibility to certain types of disasters.
# * Severe Storm Warnings: The frequency of severe storm warnings does not vary across regions and cities.

#%%

# Data Imbalance check

happening_counts = df1['happening'].value_counts()
print(happening_counts)

# Plotting the distribution of the 'happening' variable
plt.figure(figsize=(6, 4))
sns.countplot(x='happening', data=df1)
plt.title('Distribution of Happening Variable')
plt.xlabel('Happening')
plt.ylabel('Count')
plt.show()

# Calculating the imbalance ratio
imbalance_ratio = happening_counts[0] / happening_counts[1]
print("Imbalance Ratio:", imbalance_ratio)


# %%[markdown]
#This indicates that the majority class (1.0) has approximately twice as many instances as the minority class (0.0)

# %%


# Question 1: Trend of global temperature changes vs. US opinion of climate change
plt.figure(figsize=(12, 6))
sns.lineplot(data=df1, x='year', y='g_temp', label='Global Temperature')
sns.lineplot(data=df1, x='year', y='happening', label='US Opinion of Climate Change')
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Global Temperature Changes vs. US Opinion of Climate Change')
plt.legend()
plt.show()
#%%
# Question 2: Relationship between temperature/rainfall and belief in climate change
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df1, x='rainfall', y='c_temp',hue = 'happening')
plt.xlabel('Rainfall')
plt.ylabel('Temperature')
plt.title('Temperature and Rainfall vs. Belief in Climate Change')
plt.legend(title='happening')
plt.show()


# %% [markdown]
# # Modeling
#%%
# Check proportions of missing values
col = df1.columns.to_list()

# %%
missing1 = df1['year'].isna()
df2 = df1[missing1]

# %%
cleaning = df1[-missing1]

#%%
# Our final dataset will be a copy of the cleaned dataset
df = cleaning.copy(deep = True)

df['month'] = df['wave'].str[0:3]
df = df.drop(['wave'], axis = 1)

# %%
# Let's check the columsn in df
df.columns
#%%
# Now that missing values have been treated, we will dive more in depth to the summary statistics of each variable
# Beginning with numeric variables:
num_cols = ['age', 'c_temp', 'snowfall', 'rainfall', 'disasters', 'storms', 'spending', 'el_nino', 'g_temp', 'g_temp_lowess', 'children', 'adults', 'population']
# Case_ID is not on this list, as it seems to be another index column that does not provide any useful information
cat_cols = ['happening', 'female', 'education', 'income', 'race', 'ideology', 'party', 'religion', 'marit_status', 'employment', 'City', 'year', 'month', 'region9']

# Select the desired columns from the original DataFrame
desired_columns = [col for col in num_cols + cat_cols if col not in ['education', 'marit_status', 'employment']]
df_clean = df.loc[:, desired_columns].copy()
print(df_clean['region9'].unique())


#%%
# Binary reduction for specific categorical columns
binary_reductions = {
    'religion': ['Other Christian', 'Catholic'],
    'race': ['White, Non-Hispanic'],
    'ideology': ['Somewhat conservative', 'Very conservative'],
    'party': ['Democrat']
}

for col, cat_vals in binary_reductions.items():
    if col in df_clean.columns:
        df_clean.loc[:, col] = df_clean[col].isin(cat_vals).astype(int)
    else:
        print(f"Column '{col}' not found in the DataFrame. Skipping binary reduction.")

print(df_clean)
df = df_clean.copy(deep=True)
# %%[markdown]
# It seems that 1 row of the income variable was incorrectly entered and only kept the last two digits instead of the full category (99 instead of $35,000 to $39,999 for example).  
# We will replace this value with the mode of income
# %%
incomecheck = df['income'] == "99"
df[incomecheck]['income']
# %%
mode_income = df['income'].mode()[0]
df.loc[df['income'] == "99", 'income'] = mode_income
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
# order_income = pd.CategoricalDtype(cats_income, ordered=True)
df['income_encoded'] = df['income'].apply(lambda x: 0 if x in cats_income[:11] else 1)

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
print(dummy_cat_cols)
dummy_cat_cols.remove('female')
dummy_cat_cols.remove('happening')
dummy_cat_cols.remove('year')
dummy_cat_cols.remove('education')
dummy_cat_cols.remove('marit_status')
dummy_cat_cols.remove('employment')

# Remove binary reduced columns from dummy_cat_cols
for col in binary_reductions.keys():
    if col in dummy_cat_cols:
        dummy_cat_cols.remove(col)

#%%
for cols in dummy_cat_cols:
  cols_dummy = pd.get_dummies(df[cols],
                                drop_first=True,
                                dtype = int)
  dummies.append(cols_dummy)
print(len(dummies))

df_withdummies = pd.concat([df.drop(dummy_cat_cols, axis=1), *dummies], axis=1)
df_withdummies.drop(['g_temp_lowess', 'children', 'adults'], axis=1, inplace=True)
print(df_withdummies.columns)

# %%
df_withdummies = pd.concat([df.drop(dummy_cat_cols, axis = 1), dummies[0], dummies[1], dummies[2], dummies[3]], axis = 1)
df_withdummies.drop(['g_temp_lowess', 'children', 'adults'], axis = 1, inplace = True)
#%%
df_withdummies.drop(['$12,500 to $14,999', '$15,000 to $19,999', '$20,000 to $24,999',
       '$25,000 to $29,999', '$30,000 to $34,999', '$35,000 to $39,999',
       '$40,000 to $49,999', '$5,000 to $7,499', '$50,000 to $59,999',
       '$60,000 to $74,999', '$7,500 to $9,999', '$75,000 to $84,999',
       '$85,000 to $99,999', 'Less than $5,000','$100,000 to $124,999', '$125,000 to $149,999',
       '$150,000 to $174,999',
       '$175,000 to $199,999 (Nov 2016 on); $175,000 or more (Nov 2008 - Mar 2016)',
       '$200,000 to $249,999 (Nov 2016 on)', '$250,000 or more (Nov 2016 on)', 'Chicago', 'Houston',
       'Jacksonville', 'Kansas City', 'Los Angeles', 'Nashville',
       'New York City', 'Phoenix', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
       'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], axis = 1, inplace = True)
df_withdummies.columns


# %%


#%%
# Model building function
def train_and_evaluate_model(X, y):
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    firth_model = LogisticRegressionCV(penalty='l1', solver='saga', cv=5, random_state=42, max_iter=1000)
    firth_model.fit(X_train, y_train)

    y_pred = firth_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)
    print("Coefficients:", firth_model.coef_)

#%%
# Base model with standardized the data
# Selecting female, age, education level, income, race, ideology, party, religion, year, location for the base model
X = df_withdummies.drop(['happening', 'rainfall', 'snowfall', 'population', 'el_nino', 'c_temp', 'g_temp', 'storms', 'disasters', 'spending'], axis=1)
y = df_withdummies['happening']
print('Base Model: \n')
train_and_evaluate_model(X, y)

# %%
# Q2: How has temperature and rainfall impacted the perception of climate change occurring in individuals in the US since 2000?
# base model + rainfall
print('Q2 Model with `rainfall`: \n')
X = df_withdummies.drop(['happening', 'snowfall', 'population', 'el_nino', 'g_temp', 'c_temp',
       'storms', 'disasters', 'spending'], axis=1)
train_and_evaluate_model(X, y)
#%%
# base model +  c_temp
print('Q2 Model with `c_temp`: \n')
X = df_withdummies.drop(['happening', 'rainfall', 'snowfall', 'population', 'el_nino', 'g_temp',
       'storms', 'disasters', 'spending'], axis=1)
train_and_evaluate_model(X, y)

# %%
# Q3: How has the El Nino/La Nina weather pattern impacted public perception of climate change since 2000?
# base model + el_nino
print('Q3 Model: \n')
X = df_withdummies.drop(['happening', 'rainfall', 'snowfall', 'population', 'c_temp', 'g_temp',
       'storms', 'disasters', 'spending'], axis=1)
train_and_evaluate_model(X, y)
# %%
# Q4: How have weather patterns impacted the perceptions of climate change among different political and socio-economic groups since 2000?
# base model + rainfall + el+nino + g_temp
print('Q4 Model with `rainfall`, `el_nino`, and `g_temp`: \n')
X = df_withdummies.drop(['happening', 'snowfall', 'population', 'c_temp',
       'storms', 'disasters', 'spending'], axis=1)
train_and_evaluate_model(X, y)

# base model + c_temp + el_nino
print('Q4 Model with `c_temp` and `el_nino`: \n')
X = df_withdummies.drop(['happening', 'rainfall', 'snowfall', 'population', 'g_temp',
      'storms', 'disasters', 'spending'], axis=1)
train_and_evaluate_model(X, y)

# %%
# Q5: How has extreme weather impacted public perception of climate change since 2000?
# base model + storms + disasters + spending
print('Q5 Model: \n')
X = df_withdummies.drop(['happening', 'rainfall', 'snowfall', 'population', 'el_nino', 'c_temp', 'g_temp'], axis=1)
train_and_evaluate_model(X, y)
