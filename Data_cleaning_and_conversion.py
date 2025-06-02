import pandas as pd
import arff

# Load the CSV file
df = pd.read_csv("Testing-1.csv")

# Clean 'prognosis' column: remove leading/trailing spaces, lowercase
df['prognosis'] = df['prognosis'].str.strip().str.lower()

# Remove Duplicates
df = df.drop_duplicates()

# Standardize Case in 'prognosis' (optional, but good practice)
df['prognosis'] = df['prognosis'].str.title()  # Capitalize first letter of each word

# Identify and Handle Columns with Constant Values (potential irrelevant data)
constant_cols = []
for col in df.columns:
    if df[col].nunique() == 1:  # Check if the number of unique values is 1
        constant_cols.append(col)

# Remove the Constant Columns
df = df.drop(columns=constant_cols)

# Save the cleaned DataFrame to a new CSV file
cleaned_file_path = "cleaned_disease_data.csv"  # Choose a name for your new file
df.to_csv(cleaned_file_path, index=False)  # index=False prevents writing row indices to the file

print(f"Cleaned data saved to: {cleaned_file_path}")

# Convert DataFrame to ARFF format
arff_file_path = "cleaned_disease_data.arff"  # Choose a name for your ARFF file

# Create a list of tuples for the attributes, including data types
attributes = []
for col in df.columns:
    if col == 'prognosis':
        # For the 'prognosis' column, extract unique values for nominal type
        unique_values = df['prognosis'].unique().tolist()
        attributes.append((col, unique_values))
    else:
        # Numeric attributes
        attributes.append((col, 'NUMERIC'))

# Prepare the data dictionary for the ARFF file
data_dict = {
    'relation': 'disease_prediction',  # Replace with your relation name
    'attributes': attributes,
    'data': df.values.tolist()
}

# Write the ARFF file
with open(arff_file_path, 'w') as f:
    arff.dump(data_dict, f)

print(f"Cleaned data saved to ARFF format: {arff_file_path}")
