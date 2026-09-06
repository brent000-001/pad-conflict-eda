import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="PAD Conflict Analysis", layout="wide")
st.title("📊 PAD Conflict Analysis - Final Pipeline")

df = pd.read_csv("pad_conflicts.csv")
st.write(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
st.dataframe(df.head())

# --- 1. FINAL PIPELINE ---
st.divider()
st.header("1. Train-Test Split & Pipeline")

X = df[['department','conflict_cause']]
y = df['score']

# Proper pipeline: Encode + Model
preprocess = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['department','conflict_cause'])]
)

pipeline = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', RandomForestClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("Train Size", f"{len(X_train)} ({80}%)")
col2.metric("Test Size", f"{len(X_test)} ({20}%)")
col3.metric("Accuracy", f"{acc*100:.1f}%")

st.code(f"""
Train: {len(X_train)} samples
Test: {len(X_test)} samples
Accuracy: {acc:.2f}
""")

with st.expander("Classification Report"):
    st.text(classification_report(y_test, y_pred))

# --- 2. CHARTS ---
st.divider()
st.header("2. Charts")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Conflicts by Department")
    st.bar_chart(df['department'].value_counts())
with c2:
    st.subheader("Average Score by Department")
    st.bar_chart(df.groupby('department')['score'].mean())

st.subheader("Conflict Causes Distribution")
st.bar_chart(df['conflict_cause'].value_counts())

# --- 3. PREDICTION (Using Final Pipeline) ---
st.divider()
st.header("3. 🔮 Prediction - Final Pipeline")
d1, d2 = st.columns(2)
with d1:
    dept = st.selectbox("Department", sorted(df['department'].unique()))
with d2:
    cause = st.selectbox("Conflict Cause", sorted(df['conflict_cause'].unique()))

if st.button("Predict Score", type="primary"):
    pred = pipeline.predict(pd.DataFrame([[dept, cause]], columns=['department','conflict_cause']))[0]
    st.success(f"Predicted Score: **{pred}** for {dept} - {cause}")
    if pred >= 8:
        st.warning("⚠️ High Risk")
    elif pred >= 5:
        st.info("Moderate Risk")
    else:
        st.info("Low Risk")

# --- 4. OUTLIERS ---
st.divider()
st.header("4. ⚠️ Outlier Detection")

# For outliers we need encoded version
