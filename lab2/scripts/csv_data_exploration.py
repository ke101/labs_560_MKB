import pandas as pd
import os

def explore_insurance_csv(file_path):
    """
    Function to explore the Insurance Claims CSV dataset.
    Requirements covered: 
    1. [cite_start]Load CSV [cite: 55, 63]
    2. [cite_start]Display first few records [cite: 64]
    3. [cite_start]Calculate dataset size and dimensions [cite: 64]
    4. [cite_start]Identify missing data [cite: 64]
    """
    print("--- [Step 1] Loading CSV Data ---")
    
    # Check if the file exists locally
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found. Please ensure it is in the same directory.")
        return

    # 1. Load the dataset
    try:
        df = pd.read_csv(file_path)
        print("Successfully loaded the dataset.")
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # 2. Basic Operation: Display the first 5 records
    print("\n--- [Step 2] Displaying First 5 Records ---")
    print(df.head())

    # 3. Calculate dataset size and dimensions
    print("\n--- [Step 3] Dataset Dimensions ---")
    rows, cols = df.shape
    print(f"Total Rows (Records): {rows}")
    print(f"Total Columns (Features): {cols}")
    
    # 4. Identify Missing Data
    print("\n--- [Step 4] Checking for Missing Data ---")
    missing_data = df.isnull().sum()
    # Filter to show only columns with missing values
    missing_data = missing_data[missing_data > 0]
    
    if missing_data.empty:
        print("Great! No missing values found in this dataset.")
    else:
        print("Columns with missing values and their counts:")
        print(missing_data)

    # 5. Basic Data Exploration & Statistics
    print("\n--- [Step 5] Basic Statistics & Distribution ---")
    
    # Display statistical summary for numerical columns (Mean, Std, Min, Max, etc.)
    print("Statistics for numerical columns (e.g., Claim Amounts, Premiums):")
    # specific columns relevant to insurance to keep output clean
    if 'total_claim_amount' in df.columns:
        print(df[['total_claim_amount', 'policy_annual_premium', 'months_as_customer']].describe())
    else:
        print(df.describe())

    # Example: Check the distribution of a categorical column (e.g., Incident Type)
    if 'incident_type' in df.columns:
        print("\nDistribution of Incident Types:")
        print(df['incident_type'].value_counts())

# --- Execution Entry Point ---
if __name__ == "__main__":
    # Ensure 'insurance_claims.csv' is downloaded from Kaggle and placed in the current directory
    csv_file_name = 'insurance_claims.csv' 
    explore_insurance_csv(csv_file_name)
