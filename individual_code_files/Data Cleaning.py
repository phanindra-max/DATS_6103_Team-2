# %%
# Below is the cose used to combine the different portions of data used in the Group 2 final project.
#There were four data sources used for this work.
#   -YPCCC climate survey data (https://osf.io/jw79p/)
#   -NOAANow data (https://sercc.com/noaa-online-weather/)
#   -FEMA data (https://www.fema.gov/openfema-data-page/disaster-declarations-summaries-v2)
#   -NASA data (https://climate.nasa.gov/vital-signs/global-temperature/?intent=121)
#For the NOAA data, no direct downloads were available. Rather I had to copy and paste records from the site into a single file representing the biggest cities in each region included in the YPCCC data.

#%%
import pandas as pd
import numpy as np

#%%
### Clean and format disater data
# Begin by reading in disaster declariations data
data = pd.read_csv("DisasterDeclarationsSummaries(1).csv")
# Drop data older than required years
data = data[data['fyDeclared'] > 1999]  
# Sum by state and year and disaster type
d_table = data.groupby(['state','fyDeclared']).count()
d_table = d_table.reset_index()
d_table = d_table.drop('declarationType', axis=1)

# Next read in FEMA spending
dollars = pd.read_csv("PublicAssistanceFundedProjectsDetails.csv")
# Deconstruct date and retain year
import datetime
dollars['declarationDate'] = pd.to_datetime(dollars['declarationDate'])
dollars['declarationDate'] = pd.DatetimeIndex(dollars['declarationDate']).year
#Sum by state and year
dollar_table = dollars.groupby(['stateCode','declarationDate']).sum()
dollar_table = dollar_table.reset_index()
dollar_table = dollar_table.drop('incidentType', axis=1)
# Merge into one file
FEMA = pd.merge(d_table, dollar_table,  how='outer', left_on=['state','fyDeclared'], right_on = ['stateCode','declarationDate'])
FEMA = [['state', 'fyDeclared', 'disasterNumber', 'projectAmount']]
#%%
#Read in region index 
region = pd.read_csv("Population and Region.csv")
region = region[['Rergion', 'State']]
# Merge into file
regions = pd.merge(region, FEMA, how='left', left_on=['State'], right_on=['state'])
# Drop unused data
regions = regions.drop('state', axis=1)
#sum by region and year
regions = regions.groupby(['State','fyDeclared']).sum()
#Change NA to 0
regions['disasterNumber'] = regions['disasterNumber'].fillna(0)
regions['projectAmount'] = regions['projectAmount'].fillna(0)

#%%
### Merge data
# read in survey data
# Survey data has had unused rows manually removed to reduce the file size
s_data = pd.read_csv("CCAM SPSS Data 2008-2022 (cleaned).csv")

# Read in weather data
# Weather data is "hand" compiled from NOAA and NASA sources.
w_data = pd.read_csv("Weather Data.csv")

# Merge weather and survey data
df = pd.merge(s_data, w_data, how='left', left_on=['region9'], right_on=['region9'])
# merge in disaster data
df1 = pd.merge(df, regions, how='left', left_on=['region9'], right_on=['Rergion'])
df1.to_csv('DATS 6103 Final Team 2 Data.csv')