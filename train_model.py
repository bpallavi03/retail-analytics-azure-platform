import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Load Cleaned Transactions
print("Loading processed transaction data...")
df = pd.read_csv("clean_transactions.csv")

# Convert TransactionDate to datetime
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

# 2. Define the Cutoff Date for Historical Feature Extraction vs Target Window
# Let's say the first 12 months is features, and last 6 months is target CLV spend
max_date = df['TransactionDate'].max()
feature_cutoff = max_date - pd.Timedelta(days=180)

print(f"Feature engineering window: Min date to {feature_cutoff.strftime('%Y-%m-%d')}")
print(f"Target spend (CLV) window: {feature_cutoff.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

# Split Transactions into Features-period and Target-period
tx_features = df[df['TransactionDate'] <= feature_cutoff]
tx_target = df[df['TransactionDate'] > feature_cutoff]

# 3. Calculate RFM features from the feature period
customer_features = tx_features.groupby('CustomerID').agg(
    RecencyDays=('TransactionDate', lambda x: (feature_cutoff - x.max()).days),
    Frequency=('TransactionID', 'count'),
    TotalSpend=('TransactionAmount', 'sum'),
    AvgSpend=('TransactionAmount', 'mean')
).reset_index()

# Extract Tier information (take the mode or first entry)
tiers = tx_features.groupby('CustomerID')['CustomerTier'].first().reset_index()
customer_features = pd.merge(customer_features, tiers, on='CustomerID')

# Convert Categorical Tier to Numerical Dummy Codes
customer_features = pd.get_dummies(customer_features, columns=['CustomerTier'], drop_first=True)

# 4. Calculate the Target Variable: Total spend in the target window (the next 180 days)
customer_target = tx_target.groupby('CustomerID')['TransactionAmount'].sum().reset_index()
customer_target.rename(columns={'TransactionAmount': 'TargetCLV'}, inplace=True)

# 5. Merge Features and Target
dataset = pd.merge(customer_features, customer_target, on='CustomerID', how='left')
dataset['TargetCLV'] = dataset['TargetCLV'].fillna(0)  # Customers who didn't buy spent 0

# Save final feature matrix for references
dataset.to_csv("clv_feature_matrix.csv", index=False)

# 6. Model Training Setup
X = dataset.drop(columns=['CustomerID', 'TargetCLV'])
y = dataset['TargetCLV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# 7. Train & Evaluate Models
best_r2 = -1
best_model = None
best_model_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"\n--- {name} Performance ---")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"R² Score (Accuracy Metric): {r2:.4f} ({r2*100:.1f}%)")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

# Save the best model
if best_model is not None:
    print(f"\nSaving the best model ({best_model_name}) to 'best_clv_model.pkl'...")
    joblib.dump(best_model, 'best_clv_model.pkl')
    # Save the column structure of training features to load later
    joblib.dump(X_train.columns.tolist(), 'model_columns.pkl')