import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="PAD Conflict Analysis - Yaounde", layout="wide")
st.title("PAD Conflict Analysis - Yaounde")
st.subheader("Final Project - Conflict Severity Predictor (Model Ready)")
st.success("EDA Visualization Complete ✅ | New Predictive Module Active")

@st.cache_data
def load_data():
    df = pd.read_csv("pad_conflicts.csv")
    return df

def show_vif():
    df = load_data()
    X_numeric = pd.get_dummies(df[['department', 'conflict_cause', 'score']], drop_first=True)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_numeric.columns
    vif_data["VIF"] = [variance_inflation_factor(X_numeric.values, i) for i in range(X_numeric.shape[1])]
    st.subheader("Multicollinearity Check (VIF)")
    st.dataframe(vif_data) 

#... rest of your code, start with def train()...


def train():
    show_vif()

    # FIX LEAKAGE - create target once
    df['severity'] = pd.cut(df['score'], bins=[0,3,6,10], labels=["Low","Medium","High"])
    
    # Features = ONLY department + cause (NOT score)
    X = pd.get_dummies(df[['department','conflict_cause']])
    y = df['severity']

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scaling - fit ONLY on train
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Outlier detection - fit ONLY on train for no leakage
    clf = IsolationForest(contamination=0.05, random_state=42)
    clf.fit(X_train)
    df['anomaly'] = clf.fit_predict(X)

clf, cols, df = train()

# 1. OUTLIERS SECTION
st.subheader("⚠️ Outliers Detected")
st.write(f"Found {(df['anomaly']==-1).sum()} outliers out of {len(df)}")
st.dataframe(df[df['anomaly']==-1][['department','conflict_cause','score','severity']])

st.divider()

# 1b. CLASS IMBALANCE + SCALING
st.subheader("⚖️ Class Imbalance & Scaling")
st.write("Severity distribution:")
st.write(df['severity'].value_counts())
st.bar_chart(df['severity'].value_counts())
st.caption("Treatment: StandardScaler applied + class_weight='balanced' to handle minority 'High' class")

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
        pred = "High"

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
