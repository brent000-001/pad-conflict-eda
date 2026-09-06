import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest

st.title("PAD Conflict Analysis")

df = pd.read_csv("pad_conflicts.csv")

# Use score as target (since no severity column)
X = pd.get_dummies(df[['department','conflict_cause']])
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Outliers
iso = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = iso.fit_predict(X)

st.subheader("⚠️ Outliers Detected")
st.write(f"Found {(df['anomaly']==-1).sum()} outliers out of {len(df)}")
st.dataframe(df[df['anomaly']==-1][['department','conflict_cause','score']])

st.subheader("All Data")
st.dataframe(df)
