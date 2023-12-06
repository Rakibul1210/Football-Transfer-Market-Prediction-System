from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import joblib


print("@1")
data_path = 'data/final-data-22-23.csv'
df = pd.read_csv(data_path)

# Droping non-numeric columns for now .........................................................
df = df.drop(['Name', 'Transfer_fee', 'Rk'], axis=1)

# Separate features and target variable
X = df.drop('Market_value', axis=1)
y = df['Market_value']

# Identify categorical columns
categorical_cols = ['Nation', 'Pos', 'Squad', 'Comp']

# Create a column transformer for encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'
)

# Encode categorical features and concatenate with numeric features
X_encoded = preprocessor.fit_transform(X)

print("@2")
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

print("@3")
# Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("@4")
# Predictions on the testing set
y_pred = model.predict(X_test)

# Calculating error metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5

print('Mean Absolute error: ', mae)
# print('Mean squared error: ', mse)
print('Root mean squared error: ', rmse)

# Save the trained model to a file
joblib.dump(model, 'model/trained_rf_model_with_categorical.joblib')
# Save the preprocessor to a file
joblib.dump(preprocessor, 'model/preprocessor_with_categorical.joblib')

print("Model Saved")
