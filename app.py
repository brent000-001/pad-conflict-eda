import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="PAD Conflict Analysis - Yaounde", layout="wide")
st.title("📊 PAD Conflict Analysis - Yaounde")
st.write("Final Project - Conflict Severity Predictor (Model Ready)")
st.success("EDA Visualization Complete ✅ | Now Predictive Module Active")

@st.cache_data
def load_data():
    df = pd.read_csv("pad_conflicts.csv")
    return df

def train():
    df = load_data()
    # Features for anomaly detection
    features = df[['score']].copy()
    # IsolationForest - detect 5% outliers
    clf = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly'] = clf.fit_predict(features)
    df['severity'] = pd.cut(df['score'], bins=[0,3,6,10], labels=['Low','Medium','High'])
    cols = ['score']
    return clf, cols, df

clf, cols, df = train()

# 1. OUTLIERS SECTION
st.subheader("⚠️ Outliers Detected")
st.write(f"Found {(df['anomaly']==-1).sum()} outliers out of {len(df)}")
st.dataframe(df[df['anomaly']==-1][['department','conflict_cause','score','anomaly','severity']])

st.divider()

# 2. PREDICTION FORM - NEW
st.subheader("🔮 Predict Conflict Severity")

col1, col2 = st.columns(2)
with col1:
    dept = st.selectbox("Department", df['department'].unique())
    cause = st.selectbox("Conflict Cause", df['conflict_cause'].unique())
with col2:
    score = st.slider("Conflict Score", 1, 10, 5)

if st.button("Predict Severity"):
    if score <= 3:
        pred = "Low"
    elif score <= 6:
        pred = "Medium"
    else:
        pred = "High

    # Check if outlier
    is_outlier = clf.predict([[score]])[0] == -1

    st.metric("Predicted Severity", pred)
    if is_outlier:
        st.error(f"⚠️ This case is an ANOMALY! High risk in {dept}")
    else:
        st.success(f"Normal case in {dept} - {cause}")

    st.info(f"Based on: Dept={dept}, Cause={cause}, Score={score}")

st.divider()
st.caption("Brent000-001 | PAD Yaounde 2026")
