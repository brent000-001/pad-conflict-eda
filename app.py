import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="PAD Conflict Predictor", page_icon="⚖️", layout="centered")
st.title("⚖️ PAD Conflict Analysis - Yaounde")
st.markdown("**Final Project - Conflict Severity Predictor (Model Ready)**")
st.success("EDA Virtualization Complete ✅ - Now Predictive Module Active")

@st.cache_resource
def train():
    df = pd.read_csv('pad_conflicts.csv')
    df['severity'] = pd.cut(df['score'], bins=[0,3,6,10], labels=['Low','Medium','High'])
    X = pd.get_dummies(df[['department','conflict_cause']], drop_first=True)
    y = df['severity']
    cols = list(X.columns)
    clf = RandomForestClassifier(random_state=42, class_weight='balanced')
    clf.fit(X, y)
    return clf, cols, df

clf, cols, df = train()

col1, col2 = st.columns(2)
with col1:
    dept = st.selectbox("Department", sorted(df['department'].unique()))
with col2:
    cause = st.selectbox("Conflict Cause", sorted(df['conflict_cause'].unique()))

if st.button("🔮 Predict Severity", type="primary"):
    df_new = pd.DataFrame({'department':[dept], 'conflict_cause':[cause]})
    X_new = pd.get_dummies(df_new, drop_first=True)
    for c in cols:
        if c not in X_new.columns: X_new[c]=0
    X_new = X_new[cols]
    pred = clf.predict(X_new)[0]
    prob = dict(zip(clf.classes_, clf.predict_proba(X_new)[0].round(2)))

    if pred=="High": st.error(f"### 🔴 HIGH - {prob}")
    elif pred=="Medium": st.warning(f"### 🟠 MEDIUM - {prob}")
    else: st.success(f"### 🟢 LOW - {prob}")

st.divider()
st.bar_chart(df['department'].value_counts())
st.caption("Deployed from Chromebook Linux! | Brent000-001")
