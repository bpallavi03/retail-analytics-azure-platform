import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

num_records = 11000  # Generate slightly over 10k records
num_customers = 1200

# Generate Customer IDs and baseline profiles
customer_ids = [f"CUST-{i:04d}" for i in range(1, num_customers + 1)]
customer_loyalty = {cid: random.choice(["Bronze", "Silver", "Gold", "Platinum"]) for cid in customer_ids}
# Base probability of purchase and average order value (AOV) based on loyalty
customer_profiles = {}
for cid, tier in customer_loyalty.items():
    if tier == "Bronze":
        customer_profiles[cid] = {"aov": random.uniform(20, 60), "freq": random.randint(1, 5)}
    elif tier == "Silver":
        customer_profiles[cid] = {"aov": random.uniform(60, 120), "freq": random.randint(4, 10)}
    elif tier == "Gold":
        customer_profiles[cid] = {"aov": random.uniform(120, 250), "freq": random.randint(8, 20)}
    else:  # Platinum
        customer_profiles[cid] = {"aov": random.uniform(250, 600), "freq": random.randint(15, 40)}

categories = ["Electronics", "Apparel", "Home & Kitchen", "Beauty & Health", "Sports & Outdoors"]

# Generate Transactions
transactions = []
start_date = datetime(2024, 1, 1)

for i in range(num_records):
    tx_id = f"TX-{i+100000:06d}"
    cust_id = random.choice(customer_ids)
    profile = customer_profiles[cust_id]
    
    # Calculate transaction date with seasonal variations (more in Nov/Dec, less in Jan)
    days_to_add = random.randint(0, 500)  # ~1.5 years of data
    tx_date = start_date + timedelta(days=days_to_add)
    
    # Adjust amount based on category and customer profile
    category = random.choice(categories)
    base_aov = profile["aov"]
    
    # Add noise & seasonal multipliers (Holiday season Nov-Dec gets 1.25x spending boost)
    multiplier = 1.25 if tx_date.month in [11, 12] else 1.0
    amount = round(np.random.normal(base_aov, base_aov * 0.15) * multiplier, 2)
    amount = max(5.0, amount)  # Avoid negative values
    
    transactions.append({
        "TransactionID": tx_id,
        "CustomerID": cust_id,
        "CustomerTier": customer_loyalty[cust_id],
        "TransactionDate": tx_date.strftime("%Y-%m-%d %H:%M:%S"),
        "ProductCategory": category,
        "TransactionAmount": amount
    })

df = pd.DataFrame(transactions)
df.to_csv("retail_transactions.csv", index=False)
print(f"Successfully generated {len(df)} transactions and saved to 'retail_transactions.csv'.")