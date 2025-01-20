import pandas as pd

# Load the Excel file
file_path = "Top_Xbox_Games_Questions.xlsx"  # Replace with your file path
excel_data = pd.ExcelFile(file_path)

# Dictionary to store dataframes for each sheet
dataframes = {}

# Iterate through each sheet name and load it as a dataframe
for sheet_name in excel_data.sheet_names:
    dataframes[sheet_name] = excel_data.parse(sheet_name)

# Access individual dataframes
for name, df in dataframes.items():
    print(f"Sheet: {name}")
    print(df.head())  # Preview the first few rows of each dataframe
