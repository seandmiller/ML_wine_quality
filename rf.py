import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('wine_quality.csv') 

X = data.drop('quality', axis=1)
y = data['quality']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)


y_pred = rf_model.predict(X_test_scaled)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared Score: {r2:.4f}")


feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))


sample_wine = X_test.iloc[0]  
sample_wine_scaled = scaler.transform([sample_wine])
predicted_quality = rf_model.predict(sample_wine_scaled)[0]
actual_quality = y_test.iloc[0]

print('random forest model')
print(f"\nExample Prediction:")
print(f"Predicted Quality: {predicted_quality:.2f}")
print(f"Actual Quality: {actual_quality}")
print(f"Features of this wine sample:")



# import joblib
# joblib.dump(rf_model, 'random_forest_wine_model.joblib')
# joblib.dump(scaler, 'wine_scaler.joblib')
# print("\nModel and scaler saved.")