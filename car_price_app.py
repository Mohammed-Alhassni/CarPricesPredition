import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit page setup
st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("🚗 Car Price Prediction using Multiple Linear Regression")

# --- Step 1: Load Dataset ---
DATA_PATH = "Cars_Dataset.csv"

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(DATA_PATH)
st.subheader("📋 Dataset Preview")
st.dataframe(df.head())

# --- Step 2: Basic Cleaning & Feature Engineering ---
df = df.dropna(subset=['Year', 'Distance', 'Price'])
df['Car_Age'] = 2025 - df['Year']


# --- Step 3: Encode Categorical Features ---
cat_cols = ['Car Name', 'Fuel', 'Drive', 'Type']
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

st.subheader("📋 Final Dataset")
st.dataframe(df.head())

# --- Step 4: Correlation Analysis ---
st.subheader("📊 Feature Correlation with Price")

corr = df.corr(numeric_only=True)
st.dataframe(corr['Price'].sort_values(ascending=False).to_frame())

# Plot correlation heatmap
fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax_corr)
ax_corr.set_title("Correlation Heatmap of Features")
st.pyplot(fig_corr)

# --- Step 5: Define Features and Target ---
X = df[['Year', 'Distance', 'Owner', 'Fuel', 'Drive', 'Type', 'Car_Age']]
y = df['Price']

# --- Step 6: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 7: Scale Numeric Data ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 8: Train the Model ---
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --- Step 9: Evaluate Model ---
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("📈 Model Evaluation")
st.write(f"**R² Score:** {r2:.3f}")
st.write(f"**Mean Absolute Error:** {mae:,.2f}")
st.write(f"**Root Mean Squared Error:** {rmse:,.2f}")

# --- Step 10: Actual vs Predicted Visualization ---
st.subheader("🔍 Actual vs Predicted Prices")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color="blue", alpha=0.6)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Actual vs Predicted Car Prices")
st.pyplot(fig)

# --- Step 11: Feature Importance (Coefficients) ---
st.subheader("🧠 Feature Importance (Model Coefficients)")
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)
st.dataframe(coef_df)

fig2, ax2 = plt.subplots()
ax2.barh(coef_df['Feature'], coef_df['Coefficient'], color="green")
ax2.set_title("Feature Coefficients")
st.pyplot(fig2)

# --- Step 12: User Input for Prediction ---
st.subheader("🎯 Predict a New Car Price")

col1, col2, col3 = st.columns(3)
with col1:
    year = st.number_input("Year of Manufacture", 2000, 2025, 2020)
    distance = st.number_input("Distance Driven (km)", 0, 300000, 10000)
    owner = st.selectbox("Number of Owners", [1, 2, 3])
with col2:
    fuel = st.selectbox("Fuel Type", df['Fuel'].unique())
    drive = st.selectbox("Transmission", df['Drive'].unique())
    type_car = st.selectbox("Car Type", df['Type'].unique())

car_age = 2025 - year
km_per_year = distance / (car_age if car_age > 0 else 1)

input_data = pd.DataFrame([[year, distance, owner, fuel, drive, type_car, car_age]],
                          columns=X.columns)
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

st.success(f"💰 **Predicted Car Price:** ₹{prediction:,.0f}")
