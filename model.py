#%%
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from EDAcopy import df_withdummies

# check the df_withdummies
print(df_withdummies.head())
print(df_withdummies.tail())
print(df_withdummies.shape)

#%%
# Select the relevant variables for modeling
X = df_withdummies[['year', 'c_temp', 'rainfall', 'snowfall', 'el_nino', 'g_temp', 'storms', 'disasters', 'spending']]  # Independent variables
y = df_withdummies['happening']  # Dependent variable

#%%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add a constant term to the independent variables
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

#%%
# Fit the probit model
probit_model = sm.Probit(y_train, X_train_sm)
probit_results = probit_model.fit_regularized(method='l1', alpha=0.1)

# Make predictions on the testing set
y_pred_prob = probit_results.predict(X_test_sm)
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Print the model summary
print(probit_results.summary())

#%%
import matplotlib.pyplot as plt

# Get the variable names and coefficients
var_names = X_train_sm.columns[1:]  # Exclude the constant term
coef = probit_results.params[1:]

#%%
# Create a bar plot of the coefficients
plt.figure(figsize=(10, 6))
plt.bar(var_names, coef)
plt.xticks(rotation=90)
plt.xlabel('Variables')
plt.ylabel('Coefficients')
plt.title('Probit Model Coefficients')
plt.tight_layout()
plt.show()


# %%
