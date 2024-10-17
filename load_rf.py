import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


rf_model = joblib.load('random_forest_wine_model.joblib')


sample_df = pd.read_csv('sample_wine_data.csv')


scaler = StandardScaler()

scaled_input = scaler.fit_transform(sample_df)


prediction = rf_model.predict(scaled_input)

print(f"Predicted wine quality: {prediction[0]:.2f}")

# If you want to get feature importances
feature_importance = pd.DataFrame({
    'feature': sample_df.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importances:")
print(feature_importance)