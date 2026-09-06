import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest

st.set_page_config(page_title="PAD Conflict Analysis")
st.title("PAD Conflict Analysis")

df = pd.read_csv("pad_conflicts.csv")

X = pd.get_dummies(df[['department','conflict_cause']])
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
cols = X.columns.tolist()

iso = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = iso.fit_predict(X)

# --- PREDICTION FORM ---
st.header("🔮 Predict Conflict Score")
st.write("Select department and cause to predict the score")

col1, col2 = st.columns(2)
with col1:
    dept = st.selectbox("Department", sorted(df['department'].unique()))
with col2:
    cause = st.selectbox("Conflict Cause", sorted(df['conflict_cause'].unique()))

if st.button("Predict Score"):
    # create input same as training
    input_df = pd.DataFrame([[dept, cause]], columns=['department','conflict_cause'])
    input_encoded = pd.get_dummies(input_df)
    # align with training columns
    for c in cols:
        if c not in input_encoded:
            input_encoded[c] = 0
    input_encoded = input_encoded[cols]

    pred = clf.predict(input_encoded)[0]
    st.success(f"Predicted Score: **{pred}** for {dept} - {cause}")

# --- OUTLIERS ---
st.divider()
st.subheader("⚠️ Outliers Detected")
st.write(f"Found {(df['anomaly']==-1).sum()} outliers out of {len(df)}")
st.dataframe(df[df['anomaly']==-1][['department','conflict_cause','score']])

st.subheader("All Data")
st.dataframe(df)
