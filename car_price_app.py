import pandas as pd  # pandas: dataframes and CSV IO
import numpy as np  # numpy: numerical utilities (arrays, sqrt)
import streamlit as st  # streamlit: web app UI
from sklearn.model_selection import train_test_split  # split data into train/test
from sklearn.linear_model import LinearRegression  # linear regression model
from sklearn.preprocessing import LabelEncoder, StandardScaler  # encode categories, scale features
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # evaluation metrics
import matplotlib.pyplot as plt  # plotting
import seaborn as sns  # plotting (heatmap)

# Streamlit page setup
st.set_page_config(page_title="Car Price Prediction", layout="wide")  # configure page title and layout
st.title("🚗 Car Price Prediction using Multiple Linear Regression")  # main title shown in the app

# --- Step 1: Load Dataset ---
DATA_PATH = "Cars_Dataset.csv"  # relative path to dataset CSV file

@st.cache_data  # cache the result of load_data so it's not re-run every interaction
def load_data(path):
    return pd.read_csv(path)  # read CSV into a pandas DataFrame and return it

df = load_data(DATA_PATH)  # load the dataset into `df`
st.subheader("📋 Dataset Preview")  # small subtitle in the UI
st.dataframe(df.head())  # show the first few rows of the dataset to the user

# --- Step 2: Basic Cleaning & Feature Engineering ---
df = df.dropna(subset=['Year', 'Distance', 'Price'])  # remove rows missing critical columns
df['Car_Age'] = 2025 - df['Year']  # create a derived feature: car age from manufacture year


# --- Step 3: Encode Categorical Features ---
cat_cols = ['Car Name', 'Fuel', 'Drive', 'Type']  # list of categorical columns to encode
# Keep a LabelEncoder for each categorical column so we can map back and forth
encoders = {}  # dictionary to store LabelEncoder instances per column
for col in cat_cols:
    le_col = LabelEncoder()  # create a new LabelEncoder for this column
    df[col] = le_col.fit_transform(df[col].astype(str))  # fit encoder on column values and transform to integers
    encoders[col] = le_col  # store the encoder so we can inverse-transform later

# --- Extract Brand from Car Name (use original strings via inverse_transform)
# Brand is the first token of the car name string (e.g., 'Maruti Alto' -> 'Maruti')
orig_names = encoders['Car Name'].inverse_transform(df['Car Name'])  # get original string names for each encoded value
brands = [str(n).split()[0] if isinstance(n, str) and len(str(n).split())>0 else str(n) for n in orig_names]  # choose first token as brand
df['Brand'] = brands  # attach 'Brand' column to DataFrame
# Encode Brand as well so it can be used in the global model
le_brand = LabelEncoder()  # encoder for brand strings
df['Brand_enc'] = le_brand.fit_transform(df['Brand'].astype(str))  # encode brand strings to integers
encoders['Brand'] = le_brand  # save brand encoder alongside other encoders

st.subheader("📋 Final Dataset")  # subtitle before showing the final preprocessing result
st.dataframe(df.head())  # show the first rows of the cleaned / encoded DataFrame

# --- Step 4: Correlation Analysis ---
st.subheader("📊 Feature Correlation with Price")  # subtitle for correlation section

# Let user choose a brand to filter the correlation analysis
brand_filter = st.selectbox("Filter correlations by Brand (or choose 'All')", ["All"] + sorted(df['Brand'].unique()))  # UI control for brand selection

if brand_filter == "All":
    corr_df = df.corr(numeric_only=True)  # compute correlations across the whole dataset (numeric cols only)
    display_df = corr_df  # use global correlations for display
else:
    subset = df[df['Brand'] == brand_filter]  # restrict DataFrame to the chosen brand
    if len(subset) < 5:
        st.warning(f"Not enough samples for brand '{brand_filter}' to compute reliable correlations (n={len(subset)}). Showing global correlations.")  # warn if too few rows
        display_df = df.corr(numeric_only=True)  # fallback to global correlations
    else:
        display_df = subset.corr(numeric_only=True)  # compute correlations for the brand subset only

st.dataframe(display_df['Price'].sort_values(ascending=False).to_frame())  # show correlation of all numeric features with Price

# Plot correlation heatmap for the chosen dataset (global or brand subset)
fig_corr, ax_corr = plt.subplots(figsize=(10, 6))  # create matplotlib figure and axis for the heatmap
sns.heatmap(display_df, cmap="coolwarm", annot=True, fmt=".2f", ax=ax_corr)  # draw the correlation heatmap with annotations
ax_corr.set_title(f"Correlation Heatmap ({'All brands' if brand_filter=='All' else brand_filter})")  # title shows which brand is used
st.pyplot(fig_corr)  # render the matplotlib figure in Streamlit

# --- Step 5: Define Features and Target ---
# Include Brand_enc in the global feature set; brand-specific models will exclude it.
features = ['Year', 'Distance', 'Owner', 'Fuel', 'Drive', 'Type', 'Car_Age', 'Brand_enc']  # features used by global model
X = df[features]  # feature matrix (DataFrame)
y = df['Price']  # target vector (Series)

# --- Step 6: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # split data for evaluation

# --- Step 7: Scale Numeric Data for global model ---
scaler = StandardScaler()  # standard scaler instance for global model
X_train_scaled = scaler.fit_transform(X_train)  # fit scaler on training features and transform them
X_test_scaled = scaler.transform(X_test)  # transform test features using the same scaler

# --- Step 8: Train the Global Model ---
model = LinearRegression()  # create linear regression model instance
model.fit(X_train_scaled, y_train)  # fit model on scaled training data

# --- Step 8b: Train per-brand models (if enough samples) ---
from collections import defaultdict  # import defaultdict for convenience (not strictly necessary)
brand_models = {}  # dictionary to hold per-brand model+scaler+meta
MIN_SAMPLES_PER_BRAND = 20  # minimum number of rows required to train a brand-specific model
for brand in sorted(df['Brand'].unique()):
    brand_mask = df['Brand'] == brand  # boolean mask for rows matching this brand
    df_b = df.loc[brand_mask]  # subset of DataFrame for this brand
    if len(df_b) < MIN_SAMPLES_PER_BRAND:
        # skip small brands; we'll use the global model for these
        continue
    # For brand-specific model exclude Brand_enc (it's constant within the brand subset)
    brand_features = [f for f in features if f != 'Brand_enc']  # remove Brand_enc from feature list for brand models
    X_b = df_b[brand_features]  # feature matrix for the brand subset
    y_b = df_b['Price']  # target vector for the brand subset
    scaler_b = StandardScaler()  # create a scaler for the brand model
    X_b_scaled = scaler_b.fit_transform(X_b)  # fit & transform brand features
    model_b = LinearRegression()  # create brand-specific linear regression model
    model_b.fit(X_b_scaled, y_b)  # train model on brand data
    brand_models[brand] = {
        'model': model_b,  # trained model
        'scaler': scaler_b,  # scaler used for that model
        'features': brand_features,  # features order expected by the model
        'n_samples': len(df_b)  # number of rows used for training
    }  # store the trained components for later use

# --- Step 9: Evaluate Model ---
y_pred = model.predict(X_test_scaled)  # predict on the scaled test set using the global model (for evaluation)
r2 = r2_score(y_test, y_pred)  # compute R^2 metric
mae = mean_absolute_error(y_test, y_pred)  # compute mean absolute error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # compute root mean squared error

st.subheader("📈 Model Evaluation")  # subtitle for evaluation section
st.write(f"**R² Score:** {r2:.3f}")  # display R^2 score
st.write(f"**Mean Absolute Error:** {mae:,.2f}")  # display MAE
st.write(f"**Root Mean Squared Error:** {rmse:,.2f}")  # display RMSE

# --- Step 10: Actual vs Predicted Visualization ---
st.subheader("🔍 Actual vs Predicted Prices")  # subtitle for scatter plot
fig, ax = plt.subplots()  # create figure + axis
ax.scatter(y_test, y_pred, color="blue", alpha=0.6)  # scatter actual vs predicted points
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 45-degree reference line
ax.set_xlabel("Actual Price")  # x-axis label
ax.set_ylabel("Predicted Price")  # y-axis label
ax.set_title("Actual vs Predicted Car Prices")  # plot title
st.pyplot(fig)  # render figure in Streamlit

# --- Step 11: Feature Importance (Coefficients) ---
st.subheader("🧠 Feature Importance (Model Coefficients)")  # subtitle for coefficients
coef_df = pd.DataFrame({
    'Feature': X.columns,  # feature names (global features)
    'Coefficient': model.coef_  # coefficients from the trained global model
}).sort_values(by='Coefficient', ascending=False)  # sort by coefficient value for readability
st.dataframe(coef_df)  # show coefficients to the user

fig2, ax2 = plt.subplots()  # another figure for visualizing coefficients
ax2.barh(coef_df['Feature'], coef_df['Coefficient'], color="green")  # horizontal bar chart of coefficients
ax2.set_title("Feature Coefficients")  # title
st.pyplot(fig2)  # render in Streamlit

# --- Step 12: User Input for Prediction ---
st.subheader("🎯 Predict a New Car Price")  # subtitle for prediction UI

col1, col2, col3 = st.columns(3)  # create a 3-column layout for user inputs
with col1:
    year = st.number_input("Year of Manufacture", 2000, 2025, 2020)  # numeric input for year
    distance = st.number_input("Distance Driven (km)", 0, 300000, 10000)  # numeric input for distance
    owner = st.selectbox("Number of Owners", [1, 2, 3])  # select number of owners
with col2:
    # For selectboxes show the original string categories (inverse transform)
    fuel_names = list(encoders['Fuel'].inverse_transform(sorted(df['Fuel'].unique())))  # readable fuel options
    drive_names = list(encoders['Drive'].inverse_transform(sorted(df['Drive'].unique())))  # readable drive options
    type_names = list(encoders['Type'].inverse_transform(sorted(df['Type'].unique())))  # readable type options

    fuel = st.selectbox("Fuel Type", fuel_names)  # select fuel type by name
    drive = st.selectbox("Drive", drive_names)  # select drive by name
    type_car = st.selectbox("Car Type", type_names)  # select car type by name

with col3:
    # Brand selectbox (show original brand names)
    brand_names = sorted(df['Brand'].unique())  # list of brands present in the data
    brand_choice = st.selectbox("Brand", brand_names)  # UI control to choose brand

car_age = 2025 - year  # recompute car age from the user's input year
km_per_year = distance / (car_age if car_age > 0 else 1)  # safe compute km/year (avoid division by zero)

# Encode categorical inputs
fuel_enc = encoders['Fuel'].transform([fuel])[0]  # convert chosen fuel string to encoded integer
drive_enc = encoders['Drive'].transform([drive])[0]  # convert chosen drive string to encoded integer
type_enc = encoders['Type'].transform([type_car])[0]  # convert chosen car type to encoded integer

# Prepare prediction depending on whether a brand-specific model exists
if brand_choice in brand_models:
    bm = brand_models[brand_choice]  # retrieve stored brand model + scaler + meta
    feats = bm['features']  # feature ordering expected by this brand model
    # Build data in the order of feats
    row = []  # collect feature values in the correct order
    for f in feats:
        if f == 'Year':
            row.append(year)  # append year value
        elif f == 'Distance':
            row.append(distance)  # append distance value
        elif f == 'Owner':
            row.append(owner)  # append owner count
        elif f == 'Fuel':
            row.append(fuel_enc)  # append encoded fuel
        elif f == 'Drive':
            row.append(drive_enc)  # append encoded drive
        elif f == 'Type':
            row.append(type_enc)  # append encoded type
        elif f == 'Car_Age':
            row.append(car_age)  # append computed car age
        else:
            # fallback for unexpected feature
            row.append(0)
    input_data = pd.DataFrame([row], columns=feats)  # create DataFrame row in the expected feature order
    input_scaled = bm['scaler'].transform(input_data)  # scale using brand-specific scaler
    prediction = bm['model'].predict(input_scaled)[0]  # predict with brand-specific model
    used_model_info = f"brand model ({brand_choice}, n={bm['n_samples']})"  # metadata shown to user
else:
    # Use global model (includes Brand_enc)
    brand_enc = encoders['Brand'].transform([brand_choice])[0]  # encode the chosen brand for global model
    input_data = pd.DataFrame([[year, distance, owner, fuel_enc, drive_enc, type_enc, car_age, brand_enc]],
                              columns=X.columns)  # create DataFrame matching global features
    input_scaled = scaler.transform(input_data)  # scale with global scaler
    prediction = model.predict(input_scaled)[0]  # predict using global model
    used_model_info = "global model"  # metadata shown to user

st.success(f"💰 **Predicted Car Price:** ₹{prediction:,.0f}  —  used: {used_model_info}")  # show predicted price and which model was used
