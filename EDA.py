
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

# %%
#Temporal Analysis
new_df = df1.loc[:,['year','age','c_temp','snowfall','rainfall','disasters','storms','spending']]

# Aggregating data based on year
yearly_data = new_df.groupby('year').mean()

# Plotting the trends over time
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 15))

# Age
axes[0, 0].plot(yearly_data.index, yearly_data['age'], marker='o')
axes[0, 0].set_title('Average Age Over Time')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Age')

# Average Temperature
axes[0, 1].plot(yearly_data.index, yearly_data['c_temp'], marker='o')
axes[0, 1].set_title('Average Temperature Over Time')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Temperature')

# Snowfall
axes[1, 0].plot(yearly_data.index, yearly_data['snowfall'], marker='o')
axes[1, 0].set_title('Average Snowfall Over Time')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Snowfall')

# Rainfall
axes[1, 1].plot(yearly_data.index, yearly_data['rainfall'], marker='o')
axes[1, 1].set_title('Average Rainfall Over Time')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Rainfall')

# Disasters
axes[2, 0].plot(yearly_data.index, yearly_data['disasters'], marker='o')
axes[2, 0].set_title('Average Disasters Over Time')
axes[2, 0].set_xlabel('Year')
axes[2, 0].set_ylabel('Disasters')

# Storms
axes[2, 1].plot(yearly_data.index, yearly_data['storms'], marker='o')
axes[2, 1].set_title('Average Storms Over Time')
axes[2, 1].set_xlabel('Year')
axes[2, 1].set_ylabel('Storms')

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

# Geospatial Analysis

# Group by 'region9' and calculate mean for 'rainfall' and 'snowfall'
geographical_analysis_region = df1.groupby(['region9'])[['rainfall', 'snowfall','c_temp','disasters','storms']].mean().reset_index()

# Create a bar plot for mean rainfall by region9 
plt.figure(figsize=(12, 6))
sns.barplot(x='region9', y='rainfall', data=geographical_analysis_region)
plt.title('Mean Rainfall by Region')
plt.xlabel('Region')
plt.ylabel('Mean Rainfall')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a bar plot for mean snowfall by region9 
plt.figure(figsize=(12, 6))
sns.barplot(x='region9', y='snowfall', data=geographical_analysis_region)
plt.title('Mean Snowfall by Region')
plt.xlabel('Region')
plt.ylabel('Mean Snowfall')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a bar plot for mean temperature by region9 
plt.figure(figsize=(12, 6))
sns.barplot(x='region9', y='c_temp', data=geographical_analysis_region)
plt.title('Mean Temperature by Region')
plt.xlabel('Region')
plt.ylabel('Mean Temperature')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a bar plot for mean number of disasters by region9 
plt.figure(figsize=(12, 6))
sns.barplot(x='region9', y='disasters', data=geographical_analysis_region)
plt.title('Mean Disasters by Region')
plt.xlabel('Region')
plt.ylabel('Mean Disasters')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a bar plot for mean severe storm warnings by region9 
plt.figure(figsize=(12, 6))
sns.barplot(x='region9', y='storms', data=geographical_analysis_region)
plt.title('Mean Severe Storm Warnings by Region')
plt.xlabel('Region')
plt.ylabel('Mean Storms')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Group by 'city' and calculate mean for 'rainfall' and 'snowfall'

geographical_analysis_city = df1.groupby(['City'])[['rainfall', 'snowfall','c_temp','disasters','storms']].mean().reset_index()

# Create a bar plot for mean rainfall by city
plt.figure(figsize=(12, 6))
sns.barplot(x='City', y='rainfall', data=geographical_analysis_city)
plt.title('Mean Rainfall by City')
plt.xlabel('City')
plt.ylabel('Mean Rainfall')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a bar plot for mean snowfall by city
plt.figure(figsize=(12, 6))
sns.barplot(x='City', y='snowfall', data=geographical_analysis_city)
plt.title('Mean Snowfall by City')
plt.xlabel('City')
plt.ylabel('Mean Snowfall')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a bar plot for mean temperature by city 
plt.figure(figsize=(12, 6))
sns.barplot(x='City', y='c_temp', data=geographical_analysis_city)
plt.title('Mean Temperature by City')
plt.xlabel('City')
plt.ylabel('Mean Temperature')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a bar plot for mean number of disasters by city 
plt.figure(figsize=(12, 6))
sns.barplot(x='City', y='disasters', data=geographical_analysis_city)
plt.title('Mean Disasters by City')
plt.xlabel('City')
plt.ylabel('Mean Disasters')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create a bar plot for mean number of severe storm warnings by city
plt.figure(figsize=(12, 6))
sns.barplot(x='City', y='storms', data=geographical_analysis_city)
plt.title('Mean Severe Storm Warnings by City')
plt.xlabel('City')
plt.ylabel('Mean Storms')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# %%[markdown]
# Inference basis geospatial analysis
# * Temperature Variability: Cities and regions exhibit varying average temperatures, with some regions experiencing consistently higher or lower temperatures compared to others. This variability could be due to geographical location, elevation, or other local climate factors.
# * Snowfall Patterns: Similarly, there are differences in average snowfall across regions, with some areas experiencing heavier snowfall than others. This could be influenced by proximity to bodies of water, latitude, or elevation.
# * Rainfall Distribution: Rainfall patterns also vary across regions, with some areas receiving higher average rainfall than others. This could be influenced by proximity to oceans or other large bodies of water, as well as local topography.
# * Natural Disaster Frequency: The frequency of weather-related natural disasters varies across regions, with some areas experiencing more frequent disasters than others. This could be due to factors such as geographical location, climate, and susceptibility to certain types of disasters.
# * Severe Storm Warnings: Similarly, the frequency of severe storm warnings varies across regions, with some areas experiencing more frequent warnings than others. This could be due to local climate conditions and susceptibility to severe weather events.

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


