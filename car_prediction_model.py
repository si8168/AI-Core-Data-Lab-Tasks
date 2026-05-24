import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import Machine Learning tools from Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib # Used for saving and loading completed AI models

print("=====================================================")
print("  🚗 AI LAB: CAR PURCHASE AMOUNT PREDICTION MODEL    ")
print("=====================================================\n")

# =====================================================================
# PART 1: ANALYSIS & PREPROCESSING OF DATASET
# =====================================================================

print("--- 1. Importing & Auditing Dataset ---")
# Task: Import the real dataset from your local Excel file
# (We use openpyxl automatically behind the scenes to read the Excel sheets)
file_path = "Car_Purchasing_Data.xlsx"
car_df = pd.read_excel(file_path, engine="openpyxl")

# Task: Display first 5 rows of the dataset
print("First 5 Rows:\n", car_df.head(5))

# Task: Display last 5 rows of the dataset
print("\nLast 5 Rows:\n", car_df.tail(5))

# Task: Determine shape of the dataset (total numbers of rows and columns)
print(f"\nDataset Dimensions (Rows, Columns): {car_df.shape}")

# Task: Display concise summary of the dataset (info)
print("\nConcise Information Summary:")
car_df.info()

# Task: Check the null values in dataset (isnull)
print("\nNull Values Per Column:\n", car_df.isnull().sum())

# Task: Identify library to plot graph to understand relations among various columns
print("\n📊 Selected Charting Library: Matplotlib.pyplot is identified for analytical mapping.")

print("\n--- 2. Isolating Inputs and Outputs ---")
# Task: Create input dataset from original dataset by dropping irrelevant features
# We drop Customer Name, Email, Country because they are text fields, and drop the Target variable.
X = car_df.drop(["Customer Name", "Customer e-mail", "Country", "Car Purchase Amount"], axis=1)
print("Input Features (X) Columns:", X.columns.tolist())

# Task: Create output dataset from original dataset (Target variable)
y = car_df["Car Purchase Amount"].values.reshape(-1, 1)
print("Output Vector (y) Shape:", y.shape)

print("\n--- 3. Feature Scaling & Weight Scaling ---")
# Task: Transform input dataset into percentage based weighted between 0 and 1
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# Task: Transform output dataset into percentage based weighted between 0 and 1
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

# Task: Print first few rows of scaled input dataset
print("Scaled Inputs Sample (First 3 Rows):\n", X_scaled[:3])

# Task: Print first few rows of scaled output dataset
print("Scaled Outputs Sample (First 3 Rows):\n", y_scaled[:3])

print("\n--- 4. Training and Testing Splitting ---")
# Task: Split data into training and testing sets (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Task: Print shape of test and training data
print(f"X_train Shape: {X_train.shape} | y_train Shape: {y_train.shape}")
print(f"X_test Shape: {X_test.shape}   | y_test Shape: {y_test.shape}")

# Task: Print first few rows of test and training data
print("\nTraining Inputs Matrix Sample (First 2 Rows):\n", X_train[:2])
print("Testing Inputs Matrix Sample (First 2 Rows):\n", X_test[:2])


# =====================================================================
# PART 2: TRAINING AND EVALUATION
# =====================================================================

print("\n--- 5. Model Initialization & Training ---")
# Task: Import and initialize AI models & Create an instance of each model 
model_lr = LinearRegression()
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Task: Train models using training data sets (input and output) with fit()
model_lr.fit(X_train, y_train.ravel())
model_rf.fit(X_train, y_train.ravel())
print("✅ Success: Linear Regression & Random Forest models are fully trained.")

print("\n--- 6. Test Data Prediction & Metric Evaluation ---")
# Task: Prediction on test data using predict() against input test data only
pred_lr_scaled = model_lr.predict(X_test)
pred_rf_scaled = model_rf.predict(X_test)

# In order to measure real dollars accuracy, we convert scaled results back to normal form
pred_lr = scaler_y.inverse_transform(pred_lr_scaled.reshape(-1, 1))
pred_rf = scaler_y.inverse_transform(pred_rf_scaled.reshape(-1, 1))
y_test_dollars = scaler_y.inverse_transform(y_test)

# Task: Evaluate model performance using mean_squared_error() & Display results (RMSE)
mse_lr = mean_squared_error(y_test_dollars, pred_lr)
rmse_lr = np.sqrt(mse_lr)

mse_rf = mean_squared_error(y_test_dollars, pred_rf)
rmse_rf = np.sqrt(mse_rf)

print(f"📊 Linear Regression Model Performance (RMSE): ${rmse_lr:.2f}")
print(f"📊 Random Forest Regressor Performance (RMSE): ${rmse_rf:.2f}")


# =====================================================================
# PART 3: BEST MODEL VISUALIZATION & MODEL STORAGE
# =====================================================================

print("\n--- 7. Selecting and Visualizing the Optimal Model ---")
# Task: Choose best model (The one with the lower error/RMSE value)
best_model_name = "Linear Regression" if rmse_lr < rmse_rf else "Random Forest Regressor"
best_model_instance = model_lr if rmse_lr < rmse_rf else model_rf
print(f"🏆 Best Selected Model Based on Low Error Metrics: {best_model_name}")

# Task: Visualize model results by creating a bar chart & Add RMSE values on top of bars
models = ['Linear Regression', 'Random Forest']
rmse_values = [rmse_lr, rmse_rf]

plt.figure(figsize=(6, 5))
bars = plt.bar(models, rmse_values, color=['#4285F4', '#34A853'], edgecolor='black', width=0.5)
plt.title("AI Model Prediction Error Comparison (Lower is Better)")
plt.ylabel("Root Mean Squared Error (RMSE) in $")

# Loop to put value metrics text directly on top of each bar chart block
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"${yval:.2f}", ha='center', va='bottom', fontweight='bold')

# Task: Display chart & Save Figure
plt.tight_layout()
plt.savefig("model_error_comparison.png")
print("💾 Performance Bar Chart saved to disk as 'model_error_comparison.png'")
plt.close()

print("\n--- 8. Saving and Loading the Model Asset ---")
# Task: Save the model to a file
model_filename = "best_car_predictor_model.pkl"
joblib.dump(best_model_instance, model_filename)
print(f"💾 Successfully exported the trained model binary asset to: '{model_filename}'")

# Task: Load the model back into program memory
loaded_car_ai = joblib.load(model_filename)
print("📂 Successfully loaded the model back into active memory.")


# =====================================================================
# PART 4: CUSTOM INPUT PREDICTION PREDICTOR
# =====================================================================

print("\n--- 9. Live Interactive Prediction Loop Simulator ---")
# Task: Gather user inputs representing a custom new client profile
# Variables sequence match: [Gender (1=Male, 0=Female), Age, Annual Salary, Credit Card Debt, Net Worth]
custom_client_input = np.array([[0, 42, 62000, 11000, 240000]]) 
print(f"Incoming Custom Client Data Matrix: {custom_client_input}")

# Scale the custom client inputs using our fitted MinMaxScaler parameters
custom_input_scaled = scaler_X.transform(custom_client_input)

# Task: Use model to make predictions based on user input / Predict on new test data
predicted_output_scaled = loaded_car_ai.predict(custom_input_scaled)

# Convert the scaled percentage output value back into real financial dollars amount
predicted_purchase_dollars = scaler_y.inverse_transform(predicted_output_scaled.reshape(-1, 1))

print(f"\n🚀 AI PREDICTION RESULT:")
print(f"   Based on the data profile variables, the target client is predicted")
print(f"   to spend exactly: ${predicted_purchase_dollars[0][0]:,.2f} on their vehicle purchase.")
print("=====================================================")