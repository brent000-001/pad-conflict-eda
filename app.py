import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest

st.title("PAD Conflict Analysis")

@st.cache_data
def load_data():
    df = pd.read_csv("conflict_data.csv") # change to your csv name
    return df

@st.cache_resource
def train():
    df = load_data()
    X = pd.get_dummies(df[['department','conflict_cause']])
    y = df['severity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf, X.columns.tolist(), df

clf, cols, df = train()

iso = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = iso.fit_predict(pd.get_dummies(df[['department','conflict_cause']]))

# 1. OUTLIERS SECTION
st.subheader("⚠️ Outliers Detected")
st.write(f"Found {(df['anomaly']==-1).sum()} outliers out of {len(df)}")
st.dataframe(df[df['anomaly']==-1][['department','conflict_cause','score','severity']])
