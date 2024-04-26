#%%
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegressionCV
import seaborn as sns
import pandas as pd
import numpy as np
# %%
# import the dataset
df1 = pd.read_csv('DATS 6103 Final Team 2 Data.csv')
df1.head()
# Unnamed:, Case_ID, and region9 contain redundant index values, so they can be dropped
df1 = df1.drop(["Unnamed: 0", "case_ID"], axis = 1)
#%%
# Check proportions of missing values
col = df1.columns.to_list()

# %%
missing1 = df1['year'].isna()
df2 = df1[missing1]

# %%
cleaning = df1[-missing1]


# %%
missing2 = cleaning['marit_status'].isna()

df3 = cleaning[missing2]


#%%
# Our final dataset will be a copy of the cleaned dataset
df = cleaning.copy(deep = True)

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
import statistics
df.iloc[30041, 5] = statistics.mode(df['income'])

#%%
spendingcheck = df['spending'] > 5000
df[spendingcheck]['spending']
# It seems like there are genuinely several rows with high spending, and this variable is just very thinly distributed with a massive spike at 0. 
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
#%%
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


df_withdummies["Conservative"] = np.where((df_withdummies['Somewhat conservative'] == 1) | (df_withdummies['Very conservative'] == 1), 1, 0)
df_withdummies.drop(['$12,500 to $14,999', '$15,000 to $19,999', '$20,000 to $24,999',
       '$25,000 to $29,999', '$30,000 to $34,999', '$35,000 to $39,999',
       '$40,000 to $49,999', '$5,000 to $7,499', '$50,000 to $59,999',
       '$60,000 to $74,999', '$7,500 to $9,999', '$75,000 to $84,999',
       '$85,000 to $99,999', 'Less than $5,000','$100,000 to $124,999', '$125,000 to $149,999',
       '$150,000 to $174,999',
       '$175,000 to $199,999 (Nov 2016 on); $175,000 or more (Nov 2008 - Mar 2016)',
       '$200,000 to $249,999 (Nov 2016 on)', '$250,000 or more (Nov 2016 on)', 'Living with partner', 'Married', 'Never married',
       'Separated', 'Widowed', 'Working - self-employed',
       'Not working - on temporary layoff from a job',
       'Not working - looking for work', 'Not working - disabled',
       'Not working - retired', 'Not working - other', 'Chicago', 'Houston',
       'Jacksonville', 'Kansas City', 'Los Angeles', 'Nashville',
       'New York City', 'Phoenix', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
       'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], axis = 1, inplace = True)
df_withdummies.columns

# %%
# Base model with standardized the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Selecting female, age, education level, income, race, ideology, party, religion, year, location for the base model
X = df_withdummies.drop(['happening', 'rainfall', 'children', 'snowfall', 'adults', 'population', 'el_nino', 'c_temp', 'g_temp',
       'g_temp_lowess', 'storms', 'disasters', 'spending'], axis=1)
X = scaler.fit_transform(X)
y = df_withdummies['happening']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Firth's logistic regression model
firth_model = LogisticRegressionCV(penalty='l1', solver='saga', cv=5, random_state=42, max_iter=1000)
firth_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = firth_model.predict(X_test)

# Calculate accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
# report the classification metrics of only second class
print("Classification Report:\n", class_report[1,0])
print("Coefficients:", firth_model.coef_)
# %% [markdown]
# So, now we have the base model which predicts at 0.733 accuracy. 
# Now, let's try to answer the SMART goal questions one by one.

# %%
# Q1: How have global temperature changes impacted US opinion of the existence of climate change since 2000?
# Base model + g_temp
X = df_withdummies.drop(['happening', 'rainfall', 'children', 'snowfall', 'adults', 'population', 'el_nino', 'c_temp',
       'g_temp_lowess', 'storms', 'disasters', 'spending'], axis=1)

X.columns
#%%
X = scaler.fit_transform(X)
y = df_withdummies['happening']

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

# %%
# Q2: How has temperature and rainfall impacted the perception of climate change occurring in individuals in the US since 2000?
# base model + rainfall
X = df_withdummies.drop(['happening', 'children', 'snowfall', 'adults', 'population', 'el_nino', 'g_temp', 'c_temp',
       'g_temp_lowess', 'storms', 'disasters', 'spending'], axis=1)
X.columns
#%%
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

# base model +  c_temp
X = df_withdummies.drop(['happening', 'rainfall', 'children', 'snowfall', 'adults', 'population', 'el_nino', 'g_temp',
       'g_temp_lowess', 'storms', 'disasters', 'spending'], axis=1)
X.columns
#%%
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


# %%
# Q3: How has the El Nino/La Nina weather pattern impacted public perception of climate change since 2000?
# base model + el_nino
X = df_withdummies.drop(['happening', 'rainfall', 'children', 'snowfall', 'adults', 'population', 'c_temp', 'g_temp',
       'g_temp_lowess', 'storms', 'disasters', 'spending'], axis=1)
X.columns
#%%
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
print("Classification Report:\n",
class_report)
print("Coefficients:", firth_model.coef_)

# %%
# Q4: How have weather patterns impacted the perceptions of climate change among different political and socio-economic groups since 2000?
# base model + rainfall + el+nino + g_temp
X = df_withdummies.drop(['happening', 'children', 'snowfall', 'adults', 'population', 'c_temp',
       'g_temp_lowess', 'storms', 'disasters', 'spending'], axis=1)
X.columns
#%%
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

# base model + c_temp + el_nino
X = df_withdummies.drop(['happening', 'rainfall', 'children', 'snowfall', 'adults', 'population', 'g_temp',
       'g_temp_lowess', 'storms', 'disasters', 'spending'], axis=1)
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



# %%
# Q5: How has extreme weather impacted public perception of climate change since 2000?
# base model + storms + disasters + spending
X = df_withdummies.drop(['happening', 'rainfall', 'children', 'snowfall', 'adults', 'population', 'el_nino', 'c_temp', 'g_temp',
       'g_temp_lowess'], axis=1)
X.columns
#%%
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
print("Signi")

# %%
