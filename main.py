from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
import pandas as pd



data_path = 'data\data_22-23.csv'
df = pd.read_csv(data_path)



# Droping non-numeric colummns for now .........................................................
df = df.drop(['Name', 'Nation', 'Pos', 'Squad', 'Comp', 'Transfer_fee','Rk'], axis=1)

print(df.dtypes)


# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Market_value']), df['Market_value'], test_size=0.2, random_state=42)

# print(X_train.dtypes)
# print(y_train.dtypes)
# print(X_test.dtypes)
# print(y_test.dtypes)



# Training 
model = RandomForestRegressor(n_estimators=100, random_state=42) 
model.fit(X_train, y_train)

# Predictions on the testing set
y_pred = model.predict(X_test)

# Calculate error metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5

print('Mean Absolute error: ', mae)
print('Mean squared error: ', mse) 
print('Root mean squared error: ', rmse)

# r2 = r2_score(y_test, y_pred)
# r2_percentage = r2 * 100

# print(f"R-squared: {r2_percentage:.2f}%")
