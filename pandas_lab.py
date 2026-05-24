import pandas as pd
import numpy as np

print("=========================================")
print("          📌 PANDAS CORE LAB             ")
print("=========================================\n")

# ----------------------------------------------------
# 1. DataFrame Creation and Inspection
# ----------------------------------------------------
print("--- 1. Creation & Inspection ---")
# Task: Create a DataFrame from a dictionary of lists
student_data = {
    "StudentID": [101, 102, 103, 104, 105],
    "Name": ["Amelia", "Liam", "Tevita", "Chloe", "Mānu"],
    "Score": [85, np.nan, 92, 78, 95],
    "Attempts": [1, 2, 1, 3, 1]
}
df = pd.DataFrame(student_data)
print("Initial DataFrame:\n", df)

# Task: Load a CSV file into a DataFrame and display first 5 rows
# (We will save our dataframe as a CSV first so we have one to load!)
df.to_csv("students.csv", index=False)
df_loaded = pd.read_csv("students.csv")
print("\nLoaded CSV (First 5 Rows):\n", df_loaded.head(5))

# Task: Display summary statistics using describe()
print("\nSummary Statistics:\n", df_loaded.describe())

# Task: Check for missing values
print("\nMissing Values Check:\n", df_loaded.isna().sum())


# ----------------------------------------------------
# 2. Data Selection and Filtering
# ----------------------------------------------------
print("\n--- 2. Selection & Filtering ---")
# Task: Select a specific column
print("Selected Score Column:\n", df_loaded["Score"])

# Task: Filter rows based on a condition (Score > 80)
high_scorers = df_loaded[df_loaded["Score"] > 80]
print("\nFiltered Rows (Score > 80):\n", high_scorers)

# Task: Select a subset of rows and columns using iloc and loc
# loc uses labels: Rows index 0 to 2, columns 'Name' and 'Score'
print("\nSubset using .loc:\n", df_loaded.loc[0:2, ["Name", "Score"]])
# iloc uses integers: Row index 4, Column index 1 ('Name')
print("\nSubset using .iloc (Row 4, Col 1):", df_loaded.iloc[4, 1])

# Task: Add a new column resulting from a mathematical operation
df_loaded["Total_Points"] = df_loaded["Score"] * df_loaded["Attempts"]
print("\nDataFrame with New Calculated Column:\n", df_loaded)


# ----------------------------------------------------
# 3. Data Cleaning and Manipulation
# ----------------------------------------------------
print("\n--- 3. Cleaning & Manipulation ---")
# Task: Fill missing values in a DataFrame with the mean
score_mean = df_loaded["Score"].mean()
df_filled = df_loaded.fillna({"Score": score_mean})
print("DataFrame after Filling Missing Values with Mean:\n", df_filled)

# Task: Group data by a column and compute aggregate statistics
grouped_summary = df_filled.groupby("Attempts")["Score"].mean()
print("\nGrouped Summary (Mean Score by Attempt Count):\n", grouped_summary)

# Task: Merge two DataFrames based on a common column
extra_info = pd.DataFrame({
    "StudentID": [101, 102, 103, 104, 105],
    "Subject": ["Math", "English", "Math", "Science", "English"]
})
df_merged = pd.merge(df_filled, extra_info, on="StudentID")
print("\nMerged DataFrames:\n", df_merged)

# Task: Pivot a DataFrame to create a pivot table
pivot_table = df_merged.pivot_table(values="Score", index="Subject", columns="Attempts", aggfunc="mean")
print("\nPivot Table (Mean Score by Subject & Attempts):\n", pivot_table)