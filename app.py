import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest

st.set_page_config(page_title="PAD Conflict Predictor", page_icon="⚖️", layout="centered")
st.title("⚖️ PAD Conflict Analysis - Yaounde")
st.markdown("Final Project - Conflict Severity Predictor (Model Ready)")
st.success("EDA Visualization Complete ✅ - Now Predictive Module Active")

@st.cache_data
def train():
    df = pd.read_csv('pad_conflicts.csv')
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) > 0:
        iso = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly'] = iso.fit_predict(df[numeric_cols].fillna(0))
    else:
        df['anomaly'] = 1
    df['severity'] = pd.cut(df['score'], bins=[0,5,8,10], labels=['Low','Medium','High'])
    X = pd.get_dummies(df[['department','conflict_cause']], drop_first=True)
    y = df['severity']
    cols = list(X.columns)
    clf = RandomForestClassifier(random_state=42, class_weight='balanced')
    clf.fit(X, y)
    return clf, cols, df

clf, cols, df = train()

st.subheader("⚠️ Outliers Detected")
st.write(f"Found {(df['anomaly']==-1).sum()} outliers out of {len(df)}")
st.dataframe(df[df['anomaly']==-1])
