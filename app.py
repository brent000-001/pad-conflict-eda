import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest

st.title("PAD Conflict Analysis")

# CHANGE THIS TO YOUR REAL CSV NAME
import os
st.write("Files in repo:", os.listdir(".")) # will show real csv name

# try both possible names
import pathlib
for name in ["pad-conflicts.csv", "pad_conflicts.csv", "PAD-conflicts.csv", "pad-conflicts-eda.csv"]:
    if pathlib.Path(name).exists():
        df = pd.read_csv(name)
        st.write(f"Using file: {name}")
        break
y = df['severity']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
cols = X.columns.tolist()

iso = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = iso.fit_predict(pd.get_dummies(df[['department','conflict_cause']]))

st.subheader("⚠️ Outliers Detected")
st.write(f"Found {(df['anomaly']==-1).sum()} outliers out of {len(df)}")
st.dataframe(df[df['anomaly']==-1])
