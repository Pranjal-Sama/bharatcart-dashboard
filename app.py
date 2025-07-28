import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import bcrypt
import os
from dotenv import load_dotenv

# Import necessary scikit-learn components
# Removed train_test_split as models are loaded pre-trained and not re-trained on uploaded data
# Removed direct LogisticRegression and RandomForestClassifier imports for fitting, as they are part of the joblib models
from sklearn.metrics import (
    accuracy_score, roc_auc_score, balanced_accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
# Removed imblearn.over_sampling.SMOTE as it's part of the pre-trained model's training process
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# --- GLOBAL SETUP ---
st.set_page_config(page_title="BharatCart Dashboard", layout="wide")

# --- Secure Login System ---
load_dotenv()
USER_CREDENTIALS = {"admin": os.getenv("ADMIN_HASH")}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def login():
    st.markdown("<h1 style='text-align: center;'>🔐 Login to <span style='color:#F63366;'>BharatCart</span> Dashboard</h1>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] and check_password(password, USER_CREDENTIALS[username]):
            st.success("Login successful!")
            st.session_state.logged_in = True
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# --- GLOBAL FEATURE DEFINITIONS ---
# These are the features the SDP model expects as input, AFTER preprocessing (encoding and scaling)
FEATURE_COLS_SDP = [
    'total_spent', 'procured_quantity', 'discount_percent',
    'stock_to_order_ratio', 'delivery_time_days', 'city_order_count',
    'income_bracket', 'age_bin', 'gender'
]
TARGET_COL_SDP = 'stock_deficit_flag'

# Define categorical features that need Label Encoding
LABEL_COLS_FOR_TRANSFORMATION = ['gender', 'age_bin', 'income_bracket']
# Define numerical features that need Standard Scaling
NUMERIC_COLS_FOR_SCALING = [
    'total_spent', 'discount_percent', 'stock_to_order_ratio',
    'procured_quantity', 'delivery_time_days', 'city_order_count'
]
# Define the order of features after transformation (numeric then encoded categorical)
# This order must match the order of features the pre-trained models expect
ORDERED_FEATURES_FOR_MODEL_INPUT = NUMERIC_COLS_FOR_SCALING + LABEL_COLS_FOR_TRANSFORMATION


# --- CACHED DATA LOADING & MODEL LOADING ---
@st.cache_data
def load_segmentation_data_and_fit_transformers():
    """
    Loads segmentation data and fits LabelEncoders and StandardScaler.
    These fitted transformers are then used for preprocessing single-input predictions
    for both Customer Segmentation and Stock Deficit Prediction.
    """
    try:
        df_seg = pd.read_csv("customer_df_with_segments.csv")
        profile_seg = pd.read_csv("cluster_profile.csv", index_col=0)
        
        # Initialize and fit label encoders
        label_encoders = {}
        for col in LABEL_COLS_FOR_TRANSFORMATION:
            le = LabelEncoder()
            # Fit on all unique values from the segmentation data to ensure all possible labels are learned
            le.fit(df_seg[col])
            label_encoders[col] = le
        
        # Prepare data for scaler fitting: first encode categorical columns, then concatenate with numeric ones
        df_for_scaler_fit_numeric = df_seg[NUMERIC_COLS_FOR_SCALING].copy()
        df_for_scaler_fit_categorical = df_seg[LABEL_COLS_FOR_TRANSFORMATION].copy()

        for col in LABEL_COLS_FOR_TRANSFORMATION:
            df_for_scaler_fit_categorical[col] = label_encoders[col].transform(df_for_scaler_fit_categorical[col])

        # Combine processed numeric and encoded categorical features for scaler fitting
        # Ensure the order of columns here matches ORDERED_FEATURES_FOR_MODEL_INPUT
        df_for_scaler_fit = pd.concat([df_for_scaler_fit_numeric, df_for_scaler_fit_categorical], axis=1)
        df_for_scaler_fit = df_for_scaler_fit[ORDERED_FEATURES_FOR_MODEL_INPUT] # Ensure exact order

        # Initialize and fit StandardScaler
        scaler = StandardScaler()
        scaler.fit(df_for_scaler_fit)

        segment_map = {
            0: 'Budget-Conscious',
            1: 'High Discount Seekers',
            2: 'Premium Buyers'
        }
        return df_seg, profile_seg, label_encoders, scaler, segment_map
    except FileNotFoundError:
        st.error("Missing Customer Segmentation data files! Ensure 'customer_df_with_segments.csv' and 'cluster_profile.csv' are in the same directory.")
        return pd.DataFrame(), pd.DataFrame(), {}, None, {}
    except Exception as e:
        st.error(f"Error loading segmentation data or fitting transformers: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}, None, {}

@st.cache_resource
def load_models():
    """Loads the pre-trained Random Forest and KMeans models."""
    try:
        rf_model_sm_final = joblib.load("rf_model_sm_final.joblib")
        kmeans_model = joblib.load("kmeans_model.joblib")
        return rf_model_sm_final, kmeans_model
    except FileNotFoundError:
        st.error("Model files missing! Ensure 'rf_model_sm_final.joblib' and 'kmeans_model.joblib' are present.")
        return None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load data and models at app startup
customer_df_seg, cluster_profile_seg, label_encoders_seg, scaler_seg, segment_map_seg = load_segmentation_data_and_fit_transformers()
rf_model_sm_final, kmeans_model = load_models()

# Stop app if essential components are not loaded
if rf_model_sm_final is None or kmeans_model is None or scaler_seg is None or not label_encoders_seg:
    st.error("Failed to load all necessary models or data transformers. Please check file paths and previous error messages.")
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("📁 Navigation")
page = st.sidebar.radio("Go to", ["Customer Segmentation", "Stock Deficit Prediction"])
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.experimental_rerun()

# --- Customer Segmentation Page ---
if page == "Customer Segmentation":
    st.title("📊 Customer Segmentation Dashboard")
    st.markdown("""
    <p style='font-size:16px;'>
    This section helps us understand our customers better by dividing them into distinct groups or **segments** based on their shopping behaviors, spending habits, and how they react to discounts.
    By identifying these segments, BharatCart can create **more effective, personalized marketing campaigns**, offer tailored promotions, and even refine inventory planning to match specific customer needs.
    </p>
    """, unsafe_allow_html=True)

    if not customer_df_seg.empty and not cluster_profile_seg.empty:
        st.subheader("📌 Segment Overview: Who are Our Customers?")
        st.markdown("""
        This chart shows the **distribution of our customers** across each identified segment. A larger bar means more customers fall into that group. Understanding the size of each segment helps us allocate resources effectively.
        """)
        segment_counts = customer_df_seg['segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Customer Count']
        segment_counts['%'] = round((segment_counts['Customer Count'] / segment_counts['Customer Count'].sum()) * 100, 1)

        fig1, ax1 = plt.subplots(figsize=(7, 4))
        sns.barplot(x='Segment', y='Customer Count', data=segment_counts, palette='viridis', ax=ax1)
        for index, row in segment_counts.iterrows():
            ax1.text(index, row['Customer Count'] + 200, f"{row['%']}%", ha='center', fontsize=10)
        ax1.set_title("Customer Count per Segment")
        ax1.set_xlabel("Segment")
        ax1.set_ylabel("Number of Customers")
        plt.xticks(rotation=15)
        st.pyplot(fig1)
        plt.close(fig1)

        st.subheader("📊 Cluster Profile: What Makes Each Segment Unique?")
        st.markdown("""
        This **heatmap** illustrates the average characteristics of each customer segment across various features.
        - **Warm colors (reds)** indicate features where that segment tends to score **higher** than average.
        - **Cool colors (blues)** indicate features where that segment tends to score **lower** than average.
        This helps us see at a glance how segments differ in terms of spending, discount sensitivity, age, income, and more.
        """)
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.heatmap(cluster_profile_seg.T, annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
        ax2.set_title("Cluster Feature Heatmap: Average Values per Segment")
        ax2.set_xlabel("Cluster")
        ax2.set_ylabel("Features")
        st.pyplot(fig2)
        plt.close(fig2)

        st.subheader("💸 Spending vs. Discount Behavior: How do segments interact with promotions?")
        st.markdown("""
        This **scatter plot** visualizes how different customer segments behave concerning their **total spending** and the **average discount** they receive.
        - Look for distinct clusters of colors to understand if certain segments spend more but get fewer discounts, or if they are highly sensitive to promotions.
        - This insight is crucial for designing targeted discount strategies.
        """)
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=customer_df_seg,
            x='total_spent',
            y='discount_percent',
            hue='segment',
            palette='Set2',
            s=60,
            ax=ax3
        )
        ax3.set_title("Customer Segments by Spending vs Discount Sensitivity")
        ax3.set_xlabel("Total Spent")
        ax3.set_ylabel("Avg. Discount Received")
        ax3.grid(True)
        st.pyplot(fig3)
        plt.close(fig3)

        st.subheader("📦 Spending Distribution per Segment: Understanding Spending Habits")
        st.markdown("""
        This **box plot** illustrates the range and typical spending amount within each customer segment.
        - The **box** shows where the middle 50% of customers in that segment fall in terms of spending.
        - The **line inside the box** is the median spending.
        - The **'whiskers'** extend to show the full range of typical spending, and **individual points** beyond them are considered outliers (unusually high or low spenders within that segment).
        """)
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=customer_df_seg, x='segment', y='total_spent', palette='Set3', ax=ax4)
        ax4.set_title("Total Spending by Segment")
        ax4.set_xlabel("Segment")
        ax4.set_ylabel("Total Spent")
        ax4.grid(True)
        st.pyplot(fig4)
        plt.close(fig4)

        st.subheader("📋 Cluster Profile Table: Detailed Segment Data")
        st.markdown("""
        This table provides the **numeric averages** for each feature across all identified customer segments. It's a detailed look at the data behind the heatmap, allowing for precise comparison between segments.
        """)
        st.dataframe(cluster_profile_seg.style.background_gradient(cmap='coolwarm').format("{:.2f}"))

        st.subheader("🔍 Predict Segment for a New Customer")
        with st.form("segmentation_form"):
            inputs_raw_seg = {}
            col1, col2, col3 = st.columns(3)
            # Define all expected raw input features for segmentation
            input_features_for_segmentation_raw = NUMERIC_COLS_FOR_SCALING + LABEL_COLS_FOR_TRANSFORMATION
            
            for i, col in enumerate(input_features_for_segmentation_raw):
                with [col1, col2, col3][i % 3]:
                    if col in LABEL_COLS_FOR_TRANSFORMATION:
                        # For categorical inputs, use selectbox with known classes
                        try:
                            options = label_encoders_seg[col].classes_.tolist()
                            inputs_raw_seg[col] = st.selectbox(f"{col} (Categorical)", options, key=f"seg_{col}")
                        except Exception:
                            st.warning(f"Could not load classes for {col}. Please provide a numeric value for now.")
                            inputs_raw_seg[col] = st.number_input(f"{col} (Numeric/Categorical)", value=0.0, step=0.1, key=f"seg_{col}_num")
                    else:
                        inputs_raw_seg[col] = st.number_input(col, value=0.0, step=0.1, key=f"seg_{col}")
            
            submitted_seg = st.form_submit_button("Predict Segment")
            if submitted_seg:
                # Convert raw inputs to a DataFrame
                input_df_raw_seg = pd.DataFrame([inputs_raw_seg])
                
                # Apply Label Encoding for categorical features using loaded encoders
                input_df_processed_seg = input_df_raw_seg.copy()
                for col in LABEL_COLS_FOR_TRANSFORMATION:
                    if col in input_df_processed_seg.columns and col in label_encoders_seg:
                        try:
                            input_df_processed_seg[col] = label_encoders_seg[col].transform(input_df_processed_seg[col])
                        except ValueError:
                            st.error(f"Unseen label for '{col}'. Please select a valid option from the dropdown for segmentation.")
                            st.stop()
                    elif col in input_df_processed_seg.columns and col not in label_encoders_seg:
                         st.warning(f"LabelEncoder not found for '{col}'. Assuming numerical input.")

                # Reorder columns to match scaler's fit order (ORDERED_FEATURES_FOR_MODEL_INPUT)
                input_df_processed_seg = input_df_processed_seg[ORDERED_FEATURES_FOR_MODEL_INPUT]

                # Apply Scaling using the loaded scaler
                input_scaled_seg = scaler_seg.transform(input_df_processed_seg)
                
                # Predict cluster using KMeans model
                cluster_seg = kmeans_model.predict(input_scaled_seg)[0]
                segment_name = segment_map_seg[cluster_seg]
                st.write(f"Predicted Segment: **{segment_name}**")

        with st.expander("🔍 See Sample Raw Data"):
            st.markdown("**Here are the first few rows of the input data used for segmentation:**")
            st.dataframe(customer_df_seg.head(10))
    else:
        st.info("Customer segmentation data is not available. Please ensure 'customer_df_with_segments.csv' and 'cluster_profile.csv' are in the same directory.")

# --- Stock Deficit Prediction Page ---
elif page == "Stock Deficit Prediction":
    st.title("📦 Stock Deficit Prediction Dashboard")
    st.markdown("""
    <p style='font-size:16px;'>
    This section is crucial for **preventing stockouts** and optimizing our **inventory planning**. We use advanced machine learning models to predict whether a product might face a **stock deficit** in the near future. This proactive approach helps BharatCart ensure products are always available when customers want them.
    </p>
    """, unsafe_allow_html=True)

    # Clarified uploader text: app expects already preprocessed data
    st.info("Upload a CSV file containing your **preprocessed (encoded and scaled) features** for stock deficit prediction. Ensure it includes columns like `total_spent`, `procured_quantity`, etc., in their transformed state.")
    uploaded_file = st.file_uploader("📁 Upload Preprocessed CSV", type=["csv"], help="This file should contain the necessary features already transformed (e.g., encoded categorical, scaled numerical) for the model to analyze.")

    if uploaded_file:
        try:
            df_uploaded_sdp = pd.read_csv(uploaded_file)
            
            # Validate required columns are present and numeric
            missing_cols = [col for col in FEATURE_COLS_SDP if col not in df_uploaded_sdp.columns]
            if missing_cols:
                st.error(f"Uploaded CSV is missing required features for prediction: {', '.join(missing_cols)}. Please ensure all preprocessed features are present.")
                st.stop()
            
            for col in FEATURE_COLS_SDP:
                if not pd.api.types.is_numeric_dtype(df_uploaded_sdp[col]):
                    st.error(f"Column '{col}' must contain numeric (preprocessed) values. Please check your uploaded CSV.")
                    st.stop()
            
            # Select features for prediction, ensuring they are in the correct order for the model
            X_predict = df_uploaded_sdp[FEATURE_COLS_SDP]
            
            # Check if target column exists in uploaded data for performance evaluation
            y_exists = TARGET_COL_SDP in df_uploaded_sdp.columns
            y_true = df_uploaded_sdp[TARGET_COL_SDP] if y_exists else None

            # Make predictions using the pre-trained RandomForest model
            predictions_sdp = rf_model_sm_final.predict(X_predict)
            probabilities_sdp = rf_model_sm_final.predict_proba(X_predict)[:, 1]

            st.subheader("📊 Model Performance on Uploaded Data (Random Forest After SMOTE)")
            if y_exists:
                st.write(f"**Test Accuracy:** {accuracy_score(y_true, predictions_sdp):.4f}")
                st.write(f"**ROC AUC Score:** {roc_auc_score(y_true, probabilities_sdp):.4f}")
                st.write(f"**Balanced Accuracy Score:** {balanced_accuracy_score(y_true, predictions_sdp):.4f}")

                st.subheader("📋 Classification Report")
                st.text(classification_report(y_true, predictions_sdp))

                st.subheader("📉 Confusion Matrix")
                fig_cm_final, ax_cm_final = plt.subplots(figsize=(6, 5))
                ConfusionMatrixDisplay.from_predictions(y_true, predictions_sdp, cmap='Greens', ax=ax_cm_final)
                ax_cm_final.set_title("Random Forest (After SMOTE) Confusion Matrix")
                st.pyplot(fig_cm_final)
                plt.close(fig_cm_final)
            else:
                st.info("Target column 'stock_deficit_flag' not found in the uploaded CSV. Performance metrics cannot be displayed.")
            
            st.subheader("🔍 Predict Stock Deficit for a Single Order")
            st.markdown("Enter **raw** feature values below. These will be transformed by the app before prediction using the loaded preprocessing pipelines.")
            with st.form("single_prediction_form"):
                inputs_raw_sdp = {}
                col_layout = st.columns(3)
                
                # Input fields for raw features, matching the ORDERED_FEATURES_FOR_MODEL_INPUT for consistency
                # This ensures proper order for transformation later
                for i, col in enumerate(ORDERED_FEATURES_FOR_MODEL_INPUT):
                    with col_layout[i % 3]:
                        if col in LABEL_COLS_FOR_TRANSFORMATION:
                            try:
                                options = label_encoders_seg[col].classes_.tolist()
                                inputs_raw_sdp[col] = st.selectbox(f"{col} (Categorical)", options, key=f"sdp_single_{col}")
                            except Exception:
                                st.warning(f"Could not load classes for {col}. Inputting as numeric for single prediction.")
                                inputs_raw_sdp[col] = st.number_input(f"{col} (Numeric/Categorical)", value=0.0, step=0.1, key=f"sdp_single_{col}_num")
                        else:
                            inputs_raw_sdp[col] = st.number_input(col, value=0.0, step=0.1, key=f"sdp_single_{col}")
                
                submitted_single_sdp = st.form_submit_button("Predict Single Order")
                
                if submitted_single_sdp:
                    # Create a DataFrame from raw inputs
                    single_input_df_raw_sdp = pd.DataFrame([inputs_raw_sdp])

                    # Apply Label Encoding for categorical features
                    single_input_df_processed_sdp = single_input_df_raw_sdp.copy()
                    for col in LABEL_COLS_FOR_TRANSFORMATION:
                        if col in single_input_df_processed_sdp.columns and col in label_encoders_seg:
                            try:
                                single_input_df_processed_sdp[col] = label_encoders_seg[col].transform(single_input_df_processed_sdp[col])
                            except ValueError as ve:
                                st.error(f"Error encoding '{col}': {ve}. Please select a valid option for single prediction input.")
                                st.stop()
                        elif col in single_input_df_processed_sdp.columns:
                            st.warning(f"LabelEncoder not found for '{col}'. Assuming it's already numeric or not to be encoded for single prediction.")

                    # Reorder features to match the scaler's and model's expected input order
                    df_for_scaling_and_prediction_single = single_input_df_processed_sdp[ORDERED_FEATURES_FOR_MODEL_INPUT]
                    
                    # Apply Scaling
                    input_scaled_for_sdp_single = scaler_seg.transform(df_for_scaling_and_prediction_single)
                    
                    # Convert back to DataFrame with correct column names for prediction
                    input_df_final_sdp = pd.DataFrame(input_scaled_for_sdp_single, columns=ORDERED_FEATURES_FOR_MODEL_INPUT)
                    
                    # Predict using the pre-trained RF model
                    pred_single = rf_model_sm_final.predict(input_df_final_sdp)[0]
                    prob_single = rf_model_sm_final.predict_proba(input_df_final_sdp)[0, 1]
                    st.write(f"Prediction: **{'Deficit' if pred_single == 1 else 'No Deficit'}** (Probability: {prob_single:.2%})")

            st.subheader("📈 Generated Predictions for Uploaded Data")
            st.markdown("""
            Here are the **stock deficit predictions** for the data you uploaded.
            The `predicted_stock_deficit_flag` column will show `1` if a deficit is predicted, and `0` if not.
            The `deficit_probability` indicates the model's confidence in that prediction. You can download these predictions to integrate them into your inventory management systems.
            """)
            
            prediction_results_df = df_uploaded_sdp.copy()
            prediction_results_df['predicted_stock_deficit_flag'] = predictions_sdp
            prediction_results_df['deficit_probability'] = probabilities_sdp
            
            st.write("First 10 rows with predictions:")
            st.dataframe(prediction_results_df.head(10))
            
            st.download_button(
                label="Download Predictions CSV",
                data=prediction_results_df.to_csv(index=False).encode('utf-8'),
                file_name="stock_deficit_predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}. Please ensure your CSV is correctly formatted and contains all required features (and is preprocessed as expected).")
            st.info(f"Check console for full error traceback if you are running locally. Error: {e}")
            st.stop()
    else:
        st.info("👆 Please upload a CSV file with your **preprocessed data** to see predictions and model performance. This file should contain the necessary features for the model to analyze.")

    st.subheader("✅ Final Recommendation")
    st.markdown("""
    Based on our analysis, the **Random Forest model (after applying SMOTE)** is the most robust and accurate for predicting stock deficits.
    - It is particularly effective at handling situations where stock deficits are rare events, ensuring we don't miss critical warnings.
    - Its high predictive power means BharatCart can rely on these forecasts to **proactively manage inventory**, reduce stockouts, and ultimately improve customer satisfaction.
    - **Key drivers for stock deficits** to focus on are: **`stock_to_order_ratio`**, **`procured_quantity`**, and **`city_order_count`**. By monitoring and managing these factors, BharatCart can significantly reduce the risk of future stockouts.
    """)
