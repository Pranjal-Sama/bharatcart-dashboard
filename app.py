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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, balanced_accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
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
FEATURE_COLS_SDP = [
    'total_spent', 'procured_quantity', 'discount_percent',
    'stock_to_order_ratio', 'delivery_time_days', 'city_order_count',
    'income_bracket', 'age_bin', 'gender'
]
TARGET_COL_SDP = 'stock_deficit_flag'

LABEL_COLS_SEGMENTATION = ['gender', 'age_bin', 'income_bracket']
FEATURES_FOR_SCALING_SEG = [
    'total_spent', 'discount_percent', 'stock_to_order_ratio',
    'procured_quantity', 'delivery_time_days', 'city_order_count',
    'age_bin', 'income_bracket', 'gender'
]

# --- CACHED DATA LOADING & MODEL LOADING ---
@st.cache_data
def load_segmentation_data_and_fit_transformers():
    try:
        df_seg = pd.read_csv("customer_df_with_segments.csv")
        profile_seg = pd.read_csv("cluster_profile.csv", index_col=0)
        label_encoders = {}
        for col in LABEL_COLS_SEGMENTATION:
            le = LabelEncoder()
            le.fit(df_seg[col])
            label_encoders[col] = le
        scaler = StandardScaler()
        df_for_scaler_fit = df_seg[FEATURES_FOR_SCALING_SEG]
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
        st.error(f"Error loading segmentation data: {e}")
        return pd.DataFrame(), pd.DataFrame(), {}, None, {}

@st.cache_resource
def load_models():
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

customer_df_seg, cluster_profile_seg, label_encoders_seg, scaler_seg, segment_map_seg = load_segmentation_data_and_fit_transformers()
rf_model_sm_final, kmeans_model = load_models()

if rf_model_sm_final is None or kmeans_model is None:
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("📁 Navigation")
page = st.sidebar.radio("Go to", ["Customer Segmentation", "Stock Deficit Prediction"])
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.experimental_rerun()

# --- Main Dashboard Introduction ---
if page == "Home":
    st.title("Welcome to the BharatCart E-commerce Analytics Dashboard! 🚀")
    st.markdown("""
    This dashboard is designed to help BharatCart tackle key e-commerce challenges by leveraging the power of **data analytics and machine learning**. Our main goals are:
    - **Improve Operational Efficiency**: By predicting potential stockouts.
    - **Enable Personalized Marketing**: Through intelligent customer segmentation.
    - **Drive Overall Growth**: By providing actionable insights for better decision-making.

    Use the navigation panel on the left to explore different analytical modules.
    """)
    st.image("https://via.placeholder.com/800x200?text=BharatCart+Logo+or+Relevant+E-commerce+Image", caption="Driving Smarter Decisions for BharatCart")
    st.info("Please select a module from the sidebar to begin your analysis.")

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
            inputs = {}
            col1, col2, col3 = st.columns(3)
            for i, col in enumerate(FEATURES_FOR_SCALING_SEG):
                with [col1, col2, col3][i % 3]:
                    inputs[col] = st.number_input(col, value=0.0, step=0.1, key=f"seg_{col}")
            submitted = st.form_submit_button("Predict Segment")
            if submitted:
                input_df = pd.DataFrame([inputs], columns=FEATURES_FOR_SCALING_SEG)
                input_scaled = scaler_seg.transform(input_df)
                cluster = kmeans_model.predict(input_scaled)[0]
                segment = segment_map_seg[cluster]
                st.write(f"Predicted Segment: **{segment}**")

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

    uploaded_file = st.file_uploader("📁 Upload Preprocessed CSV (with encoded features)", type=["csv"], help="Upload a CSV file containing preprocessed features for stock deficit prediction. Ensure it includes columns like total_spent, procured_quantity, discount_percent, etc.")

    if uploaded_file:
        try:
            df_encoded_sdp = pd.read_csv(uploaded_file)
            missing_cols = [col for col in FEATURE_COLS_SDP if col not in df_encoded_sdp.columns]
            if missing_cols:
                st.error(f"Uploaded CSV is missing required features: {', '.join(missing_cols)}.")
                st.stop()
            if df_encoded_sdp[FEATURE_COLS_SDP].isnull().any().any():
                st.error("Uploaded CSV contains missing values.")
                st.stop()
            for col in FEATURE_COLS_SDP:
                if not pd.api.types.is_numeric_dtype(df_encoded_sdp[col]):
                    st.error(f"Column '{col}' must contain numeric values.")
                    st.stop()

            X = df_encoded_sdp[FEATURE_COLS_SDP]
            y_exists = TARGET_COL_SDP in df_encoded_sdp.columns
            y = df_encoded_sdp[TARGET_COL_SDP] if y_exists else None

            if y_exists:
                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
                # Logistic Regression with class_weight='balanced'
                log_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
                log_model.fit(X_train, y_train)
                y_pred_log = log_model.predict(X_test)
                y_prob_log = log_model.predict_proba(X_test)[:, 1]

                # Random Forest with class_weight='balanced'
                rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
                rf_model.fit(X_train, y_train)
                y_pred_rf = rf_model.predict(X_test)
                y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

                # Pre-trained Random Forest (SMOTE)
                y_pred_rf_sm_final = rf_model_sm_final.predict(X_test)
                y_prob_rf_sm_final = rf_model_sm_final.predict_proba(X_test)[:, 1]
            else:
                X_test, y_test = X, None

            st.subheader("📊 Model Performance on Uploaded Data")
            st.markdown("""
            Here, we evaluate how well our machine learning models perform in predicting stock deficits.
            We focus on metrics that are especially important when predicting rare events like stockouts.
            **Accuracy** tells us overall correctness, but **ROC AUC** and **Balanced Accuracy** are better for imbalanced datasets, as they consider both correctly identified deficits and correctly identified non-deficits more fairly.
            """)
            if y_exists:
                st.write(f"**Random Forest (Pre-trained with SMOTE) Performance:** This model is our top performer, designed to handle the challenge of having very few actual stock deficit cases in the data.")
                st.write(f"**Test Accuracy:** {accuracy_score(y_test, y_pred_rf_sm_final):.4f} - This indicates the overall percentage of correct predictions.")
                st.write(f"**ROC AUC Score:** {roc_auc_score(y_test, y_prob_rf_sm_final):.4f} - A high score here (closer to 1.0) means the model is good at distinguishing between classes, even if one class is rare.")
                st.write(f"**Balanced Accuracy Score:** {balanced_accuracy_score(y_test, y_pred_rf_sm_final):.4f} - This score provides a more fair measure when there's an imbalance in the number of stockout vs. non-stockout situations. A score near 1.0 is excellent.")

                st.subheader("📋 Classification Report (Random Forest Pre-trained with SMOTE)")
                st.markdown("""
                The **classification report** provides a detailed breakdown of the model's performance for each class:
                - **Precision**: Out of all predicted deficits, how many were actually correct?
                - **Recall**: Out of all actual deficits, how many did the model correctly identify? This is crucial for avoiding stockouts.
                - **F1-Score**: A balance between Precision and Recall.
                - **Support**: The number of actual occurrences of each class in the test data.
                You'll notice that for `class 1` (stock deficit), recall is very important.
                """)
                st.text(classification_report(y_test, y_pred_rf_sm_final))

                st.subheader("📉 Confusion Matrix (Random Forest Pre-trained with SMOTE)")
                st.markdown("""
                The **confusion matrix** visually summarizes how well the model classified the data.
                - The top-left cell shows **True Negatives** (correctly predicted no deficit).
                - The bottom-right cell shows **True Positives** (correctly predicted a deficit).
                - The other cells show **errors**: **False Positives** (predicted deficit, but no actual deficit) and **False Negatives** (predicted no deficit, but there was an actual deficit - these are the costly errors we want to minimize!).
                """)
                fig_cm_final, ax_cm_final = plt.subplots(figsize=(6, 5))
                ConfusionMatrixDisplay.from_estimator(rf_model_sm_final, X_test, y_test, cmap='Greens', ax=ax_cm_final)
                ax_cm_final.set_title("Random Forest (Pre-trained with SMOTE) Confusion Matrix")
                st.pyplot(fig_cm_final)
                plt.close(fig_cm_final)

                st.subheader("🔍 Feature Importance (Random Forest Pre-trained with SMOTE)")
                st.markdown("""
                This chart highlights the **most influential factors** the model used to predict stock deficits.
                Features with longer bars have a greater impact. Understanding these drivers allows BharatCart to focus on specific operational areas to prevent stockouts. For example, if 'stock_to_order_ratio' is high, it means this ratio is a strong indicator of a potential deficit.
                """)
                importances_sdp = rf_model_sm_final.feature_importances_
                feat_imp_df_sdp = pd.DataFrame({'Feature': X.columns, 'Importance': importances_sdp}).sort_values(by='Importance', ascending=False)
                fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
                sns.barplot(x='Importance', y='Feature', data=feat_imp_df_sdp, palette='viridis', ax=ax_imp)
                ax_imp.set_title('🔍 Feature Importance (Random Forest Pre-trained with SMOTE)')
                ax_imp.set_xlabel('Importance Score')
                ax_imp.set_ylabel('Feature')
                st.pyplot(fig_imp)
                plt.close(fig_imp)

                st.subheader("📉 Comparison of All Models' Confusion Matrices")
                st.markdown("""
                This set of confusion matrices allows for a side-by-side comparison of all the models we evaluated. Note that 'Before/After SMOTE' labels for Logistic Regression and Random Forest refer to models trained with balanced class weights to simulate SMOTE effects, except for the pre-trained Random Forest which uses SMOTE.
                """)
                fig_all_cm, axes_all_cm = plt.subplots(2, 2, figsize=(12, 10))
                fig_all_cm.suptitle("🔍 Confusion Matrices for All Models", fontsize=16)
                ConfusionMatrixDisplay.from_estimator(log_model, X_test, y_test, ax=axes_all_cm[0, 0], cmap='Blues')
                axes_all_cm[0, 0].set_title("LogReg (Before SMOTE)")
                # Placeholder for LogReg (After SMOTE) - not computed, will use static metrics in bar plot
                axes_all_cm[0, 1].set_title("LogReg (After SMOTE)")
                axes_all_cm[0, 1].text(0.5, 0.5, "Not Computed\n(Using Static Metrics)", ha='center', va='center', fontsize=12)
                ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, ax=axes_all_cm[1, 0], cmap='Greens')
                axes_all_cm[1, 0].set_title("RF (Before SMOTE)")
                ConfusionMatrixDisplay.from_estimator(rf_model_sm_final, X_test, y_test, ax=axes_all_cm[1, 1], cmap='Greens')
                axes_all_cm[1, 1].set_title("RF (After SMOTE)")
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                st.pyplot(fig_all_cm)
                plt.close(fig_all_cm)

                st.subheader("📊 Model Comparison Chart")
                st.markdown("""
                This bar chart compares the performance of all trained models based on **Test Accuracy**, **ROC AUC**, and **Balanced Accuracy**.
                A higher score is better for all these metrics. Note that 'LogReg (After SMOTE)' metrics are static from original SMOTE-based results, as this model uses balanced weights to avoid SMOTE compatibility issues.
                """)
                models_names = ["LogReg (Before SMOTE)", "LogReg (After SMOTE)", "RF (Before SMOTE)", "RF (After SMOTE)"]
                # Static metrics from original SMOTE-based plot for LogReg (After SMOTE)
                test_accuracy_scores = [accuracy_score(y_test, y_pred_log), 0.9608, accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_rf_sm_final)]
                roc_auc_scores = [roc_auc_score(y_test, y_prob_log), 0.9982, roc_auc_score(y_test, y_prob_rf), roc_auc_score(y_test, y_prob_rf_sm_final)]
                balanced_acc_scores = [balanced_accuracy_score(y_test, y_pred_log), 0.9802, balanced_accuracy_score(y_test, y_pred_rf), balanced_accuracy_score(y_test, y_pred_rf_sm_final)]
                x_pos = np.arange(len(models_names))
                bar_width = 0.25
                fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
                ax_comp.bar(x_pos - bar_width, test_accuracy_scores, width=bar_width, label='Test Accuracy')
                ax_comp.bar(x_pos, roc_auc_scores, width=bar_width, label='ROC AUC')
                ax_comp.bar(x_pos + bar_width, balanced_acc_scores, width=bar_width, label='Balanced Accuracy')
                ax_comp.set_xticks(x_pos)
                ax_comp.set_xticklabels(models_names, rotation=15, ha='right')
                ax_comp.set_ylim(0.4, 1.05)
                ax_comp.set_ylabel("Score")
                ax_comp.set_title("📊 Model Comparison (LogReg vs RF | Before & After SMOTE)")
                ax_comp.legend()
                ax_comp.grid(axis='y', linestyle='--', alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig_comp)
                plt.close(fig_comp)
            else:
                st.info("Target column 'stock_deficit_flag' not found in the uploaded CSV. We can only generate predictions for new data.")

            st.subheader("🔍 Predict Stock Deficit for a Single Order")
            with st.form("single_prediction_form"):
                inputs = {}
                col1, col2, col3 = st.columns(3)
                for i, col in enumerate(FEATURE_COLS_SDP):
                    with [col1, col2, col3][i % 3]:
                        inputs[col] = st.number_input(col, value=0.0, step=0.1, key=f"sdp_{col}")
                submitted = st.form_submit_button("Predict")
                if submitted:
                    input_df = pd.DataFrame([inputs], columns=FEATURE_COLS_SDP)
                    pred = rf_model_sm_final.predict(input_df)[0]
                    prob = rf_model_sm_final.predict_proba(input_df)[0, 1]
                    st.write(f"Prediction: **{'Deficit' if pred == 1 else 'No Deficit'}** (Probability: {prob:.2%})")

            st.subheader("📈 Generated Predictions")
            st.markdown("""
            Here are the **stock deficit predictions** for the data you uploaded.
            The `predicted_stock_deficit_flag` column will show `1` if a deficit is predicted, and `0` if not.
            The `deficit_probability` indicates the model's confidence in that prediction. You can download these predictions to integrate them into your inventory management systems.
            """)
            predictions_all = rf_model_sm_final.predict(X)
            probabilities_all = rf_model_sm_final.predict_proba(X)[:, 1]
            prediction_results_df = df_encoded_sdp.copy()
            prediction_results_df['predicted_stock_deficit_flag'] = predictions_all
            prediction_results_df['deficit_probability'] = probabilities_all
            st.write("First 10 rows with predictions:")
            st.dataframe(prediction_results_df.head(10))
            st.download_button(
                label="Download Predictions CSV",
                data=prediction_results_df.to_csv(index=False).encode('utf-8'),
                file_name="stock_deficit_predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}. Please ensure your CSV is correctly formatted and contains all required features.")
            st.info("Check console for full error traceback if you are running locally.")
            st.stop()
    else:
        st.info("👆 Please upload a CSV file with your preprocessed data to see predictions and model performance. This file should contain the necessary features for the model to analyze.")

    st.subheader("✅ Final Recommendation")
    st.markdown("""
    Based on our analysis, the **Random Forest model (pre-trained with SMOTE)** is the most robust and accurate for predicting stock deficits.
    - It is particularly effective at handling situations where stock deficits are rare events, ensuring we don't miss critical warnings.
    - Its high predictive power means BharatCart can rely on these forecasts to **proactively manage inventory**, reduce stockouts, and ultimately improve customer satisfaction.
    - **Key drivers for stock deficits** to focus on are: **`stock_to_order_ratio`**, **`procured_quantity`**, and **`city_order_count`**. By monitoring and managing these factors, BharatCart can significantly reduce the risk of future stockouts.
    """)
