import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest

st.set_page_config(page_title="PAD Conflict Analysis", layout="wide")
st.title("📊 PAD Conflict Analysis Dashboard")

df = pd.read_csv("pad_conflicts.csv")

X = pd.get_dummies(df[['department','conflict_cause']])
y = df['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
cols = X.columns.tolist()

iso = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = iso.fit_predict(X)

# --- CHARTS ---
st.header("📈 Charts")
c1, c2 = st.columns(2)

with c1:
    st.subheader("Conflicts by Department")
    dept_count = df['department'].value_counts()
    st.bar_chart(dept_count)

with c2:
    st.subheader("Average Score by Department")
    avg_score = df.groupby('department')['score'].mean().sort_values(ascending=False)
    st.bar_chart(avg_score)

st.subheader("Conflict Causes Distribution")
cause_count = df['conflict_cause'].value_counts()
st.bar_chart(cause_count)

# --- PREDICTION FORM ---
st.divider()
st.header("🔮 Predict Conflict Score")
col1, col2 = st.columns(2)
with col1:
    dept = st.selectbox("Department", sorted(df['department'].unique()))
with col2:
    cause = st.selectbox("Conflict Cause", sorted(df['conflict_cause'].unique()))

if st.button("Predict Score", type="primary"):
    input_df = pd.DataFrame([[dept, cause]], columns=['department','conflict_cause'])
    input_encoded = pd.get_dummies(input_df)
    for c in cols:
        if c not in input_encoded:
            input_encoded[c] = 0
    input_encoded = input_encoded[cols]
    pred = clf.predict(input_encoded)[0]
    st.success(f"Predicted Score: **{pred}** for {dept} - {cause}")
    if pred >= 8:
        st.warning("⚠️ High risk conflict!")
    elif pred >= 5:
        st.info("Moderate risk")

# --- OUTLIERS ---
st.divider()
st.subheader("⚠️ Outliers Detected")
st.write(f"Found **{(df['anomaly']==-1).sum()} outliers** out of {len(df)}")
st.dataframe(df[df['anomaly']==-1][['department','conflict_cause','score']], use_container_width=True)

with st.expander("Show all data"):
    st.dataframe(df, use_container_width=True)
