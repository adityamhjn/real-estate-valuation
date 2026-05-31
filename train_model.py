import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

print("1. Loading Real Estate Asset Data...")

df = pd.read_csv('train.csv')

print("2. Isolating Wealth Metrics (Features)...")
features = ['GrLivArea', 'BedroomAbvGr', 'YearBuilt','OverallQual','GarageCars']
target = 'SalePrice'


clean_df = df[features + [target]].dropna()

X = clean_df[features]
y = clean_df[target]

print("3. Training the Random Forest Algorithm...")

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

print("4. Testing the Engine...")

fake_house = [[2500, 4, 2022,8,2]]
predicted_price = model.predict(fake_house)
print(f"---> Estimated Property Value: ${predicted_price[0]:,.2f}")

print("\n5. Saving the Asset...")
joblib.dump(model, 'real_estate_brain.pkl')
print("Engine successfully saved as real_estate_brain.pkl!")