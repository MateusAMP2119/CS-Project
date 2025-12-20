import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os

# Paths
BASE_DIR = '/Users/mateuscosta/Development/python-notebooks/CS-Project.'
INPUT_FILE = os.path.join(BASE_DIR, 'data/customer_info.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data/customer_info_cleaned.csv')
CURRENT_YEAR = 2025

print(f"Reading from {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"Error: File not found at {INPUT_FILE}")
    exit(1)

# Categorization (Context from notebook)
categorical_cols = ['customer_name', 'customer_gender', 'customer_birthdate', 'loyalty_card_number']
numerical_cols = [
    'kids_home', 'teens_home', 'latitude', 'longitude',
    'lifetime_spend_groceries', 'lifetime_spend_electronics', 
    'lifetime_spend_vegetables', 'lifetime_spend_nonalcohol_drinks',
    'lifetime_spend_alcohol_drinks', 'lifetime_spend_meat',
    'lifetime_spend_fish', 'lifetime_spend_hygiene',
    'lifetime_spend_videogames', 'lifetime_spend_petfood',
    'number_complaints', 'distinct_stores_visited', 'typical_hour', 
    'lifetime_total_distinct_products', 'percentage_of_products_bought_promotion', 
    'year_first_transaction'
]

# 3. Handle Missing Values
df_clean = df.copy()
# Impute specific columns with Mode
for col in ['kids_home', 'teens_home', 'typical_hour']:
    if col in df_clean.columns:
        mode_val = df_clean[col].mode()[0]
        df_clean[col] = df_clean[col].fillna(mode_val)

# Impute spending and complaints with 0
zero_impute_cols = [
    'number_complaints', 'distinct_stores_visited', 
    'lifetime_spend_groceries', 'lifetime_spend_electronics', 
    'lifetime_spend_vegetables', 'lifetime_spend_nonalcohol_drinks', 
    'lifetime_spend_alcohol_drinks', 'lifetime_spend_meat', 
    'lifetime_spend_fish', 'lifetime_spend_hygiene', 
    'lifetime_spend_videogames', 'lifetime_spend_petfood'
]
for col in zero_impute_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna(0)

# 4. Feature Engineering
df_eng = df_clean.copy()

def calculate_age(birthdate_str):
    if pd.isna(birthdate_str):
        return np.nan
    try:
        dt = datetime.strptime(birthdate_str, '%m/%d/%Y %I:%M %p')
        return CURRENT_YEAR - dt.year
    except:
        return np.nan

if 'customer_birthdate' in df_eng.columns:
    df_eng['Age'] = df_eng['customer_birthdate'].apply(calculate_age)
    df_eng['Age'] = df_eng['Age'].fillna(df_eng['Age'].median())

if 'loyalty_card_number' in df_eng.columns:
    df_eng['has_loyalty_card'] = df_eng['loyalty_card_number'].apply(lambda x: 0 if pd.isna(x) else 1)

# Drop unused columns
cols_to_drop = ['customer_id', 'customer_name', 'customer_birthdate', 'loyalty_card_number']
df_eng = df_eng.drop(columns=cols_to_drop, errors='ignore')

# 5. Encoding and Scaling
df_proc = df_eng.copy()

# 5.1 One-Hot Encoding
if 'customer_gender' in categorical_cols and 'customer_gender' in df_proc.columns:
    df_proc = pd.get_dummies(df_proc, columns=['customer_gender'], drop_first=True)

# 5.2 Log Transformation for Skewed Features (NEW STEP)
skewed_cols = [col for col in df_proc.columns if 'lifetime_spend_' in col]
print(f"Applying log1p transformation to {len(skewed_cols)} columns...")
for col in skewed_cols:
    skew_before = df_proc[col].skew()
    df_proc[col] = np.log1p(df_proc[col])
    skew_after = df_proc[col].skew()
    print(f"  {col}: skew {skew_before:.2f} -> {skew_after:.2f}")

# 5.3 Scaling
vars_to_scale = []
# Add defined numerical columns
for col in numerical_cols:
    if col in df_proc.columns:
        vars_to_scale.append(col)

# Add engineered numerical feature 'Age'
if 'Age' in df_proc.columns:
    vars_to_scale.append('Age')
    
# Add engineered feature 'has_loyalty_card'
if 'has_loyalty_card' in df_proc.columns:
    vars_to_scale.append('has_loyalty_card')

print(f"Scaling {len(vars_to_scale)} features: {vars_to_scale}")
scaler = StandardScaler()
df_proc[vars_to_scale] = scaler.fit_transform(df_proc[vars_to_scale])

# 6. Save Output
print(f"Saving cleaned data to {OUTPUT_FILE}...")
df_proc.to_csv(OUTPUT_FILE, index=False)
print("Preprocessing completed successfully.")
