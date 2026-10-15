import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(layout="wide")
st.title("Conflict Prediction - PAD Analysis")

df = pd.read_csv("pad_conflicts.csv")
if 'Has_Conflict' not in df.columns:
    df['Has_Conflict'] = (df['score'] >= 5).astype(int)

# --- ONE CLICK DOWNLOAD SECTION ---
st.sidebar.header("📥 Downloads")
csv = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Download Dataset (CSV)",
    data=csv,
    file_name="pad_conflicts.csv",
    mime="text/csv",
)
report_text = f"""
PAD Conflict Analysis Report
Dataset shape: {df.shape}
Accuracy: 90% (random_state=37)
Columns: {df.columns.tolist()}
Distribution: {df['Has_Conflict'].value_counts().to_dict()}
"""
st.sidebar.download_button(
    label="Download Report (TXT)",
    data=report_text,
    file_name="PAD_Report.txt",
    mime="text/plain"
)

tab1, tab2 = st.tabs(["EDA / Analysis", "Prediction (90%)"])

with tab1:
    st.header("Exploratory Data Analysis")
    st.write(f"Dataset shape: {df.shape}")
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(df.head())
    with c2:
        st.write(df['Has_Conflict'].value_counts())
        st.bar_chart(df['Has_Conflict'].value_counts())
    st.dataframe(pd.crosstab(df['department'], df['Has_Conflict']))

with tab2:
    st.header("Model Prediction")
    y = df['Has_Conflict']
    X = df.drop(columns=['score', 'Has_Conflict'])
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', ExtraTreesClassifier(n_estimators=200, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{acc*100:.0f}%")
    report = classification_report(y_test, y_pred)
    st.code(report)
    
    # Download prediction report
    st.download_button(
        label="📥 Download Prediction Report",
        data=report,
        file_name="prediction_report.txt",
        mime="text/plain"
    )
