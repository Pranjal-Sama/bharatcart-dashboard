import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, balanced_accuracy_score

from sklearn.utils.validation import check_X_y
from imblearn.over_sampling import SMOTE

# === PATCH: Safe SMOTE compatible with old sklearn ===
class SafeSMOTE(SMOTE):
    def _validate_data(self, X, y):
        try:
            return super()._validate_data(X, y)
        except AttributeError:
            return check_X_y(X, y)

# === Streamlit UI ===
st.title("BharatCart Stock Deficit Prediction")
st.write("Upload your preprocessed & encoded CSV file:")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    st.subheader("Preview of Data")
    st.dataframe(df.head())

    if 'stock_deficit' not in df.columns:
        st.error("'stock_deficit' column not found in the uploaded file.")
    else:
        # Split features & target
        X = df.drop('stock_deficit', axis=1)
        y = df['stock_deficit']

        # Apply SafeSMOTE
        sm = SafeSMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)

        # Split for training
        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
        )

        # Logistic Regression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        st.subheader("📊 Model Evaluation")

        def show_metrics(model_name, y_true, y_pred):
            st.markdown(f"#### {model_name}")
            st.write("Accuracy:", accuracy_score(y_true, y_pred))
            st.write("Balanced Accuracy:", balanced_accuracy_score(y_true, y_pred))
            st.write("ROC AUC Score:", roc_auc_score(y_true, y_pred))
            st.text("Classification Report:")
            st.text(classification_report(y_true, y_pred))

        show_metrics("Logistic Regression", y_test, y_pred_lr)
        show_metrics("Random Forest", y_test, y_pred_rf)

        # Feature Importance
        st.subheader("🔍 Feature Importances (Random Forest)")
        importances = rf.feature_importances_
        feature_names = X.columns
        imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots()
        sns.barplot(data=imp_df.head(10), x="Importance", y="Feature", ax=ax)
        st.pyplot(fig)

        # Save models
        joblib.dump(rf, "rf_model.pkl")
        joblib.dump(lr, "lr_model.pkl")
        st.success("✅ Models trained and saved successfully.")
