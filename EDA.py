
#%%[markdown]
# Below are some next steps of EDA, which includes:
# * Correlation Analysis to check if and how are the variables correlated
# * Temporal Analysis to check any treds over years
# * Data Imbalance check for the Target variable
# * Data Standardization for modeling
# * Geospatial Analysis to understand geographical patterns


#%%
import pandas as pd
import numpy as np
import statistics
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import pylab as py
import geopandas as gpd

# %%
# Import data
df1 = pd.read_csv('Users/bharatkhandelwal/Desktop/DATS_6103_Team-2/DATS 6103 Final Team 2 Data.csv')

# Unnamed:, Case_ID, and region9 contain redundant index values, so they can be dropped
df1 = df1.drop(["Unnamed: 0", "case_ID"], axis = 1)

print(df1.head())
# %%
num_cols = ['age', 'c_temp', 'snowfall', 'rainfall', 'disasters', 'storms', 'spending', 'el_nino', 'g_temp', 'g_temp_lowess', 'children', 'adults', 'population']
# Case_ID is not on this list, as it seems to be another index column that does not provide any useful information
cat_cols = ['happening', 'female', 'education', 'income', 'race', 'ideology', 'party', 'religion', 'marit_status', 'employment', 'City', 'year', 'month']


#Correlation Analysis

# Calculating the correlation matrix
correlation_matrix = df1[num_cols].corr()

# Displaying the correlation matrix
print(correlation_matrix)

plt.figure(figsize=(12, 8))

# Create a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".3f")

# Add title and display the plot
plt.title('Correlation Matrix')
plt.show()
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
