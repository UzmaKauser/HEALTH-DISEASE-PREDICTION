import pandas as pd

# Load the CSV file
df = pd.read_csv("Testing-1.csv")

# 1. Check for Missing Values (NaN, empty strings, etc.)
print("Missing values before cleaning:")
print(df.isnull().sum())  # Sum of missing values per column
print("\n")

# 2. Clean 'prognosis' column: remove leading/trailing spaces, lowercase
df['prognosis'] = df['prognosis'].str.strip().str.lower()

# 3. Remove Duplicates
num_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {num_duplicates}")
df = df.drop_duplicates()
print("Shape of DataFrame after removing duplicates:", df.shape)
print("\n")

# 4. Standardize Case in 'prognosis' (optional, but good practice)
df['prognosis'] = df['prognosis'].str.title()  # Capitalize first letter of each word

# 5. Identify and Handle Columns with Constant Values (potential irrelevant data)
constant_cols = []
for col in df.columns:
    if df[col].nunique() == 1:  # Check if the number of unique values is 1
        constant_cols.append(col)

print("Columns with constant values:", constant_cols)
print("\n")

# Print cleaned data
print("Cleaned Data")
print(df)

# Checking total number of unique diseases
print("\nNumber of unique diseases:", df['prognosis'].nunique())
