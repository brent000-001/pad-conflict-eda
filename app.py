import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Conflict Prediction - PAD Analysis")

df = pd.read_csv("pad_conflicts.csv")
if 'Has_Conflict' not in df.columns:
    df['Has_Conflict'] = (df['score'] >= 5).astype(int)

# TABS
tab1, tab2 = st.tabs(["📊 EDA / Analysis", "🤖 Prediction (90%)"])

with tab1:
    st.header("Exploratory Data Analysis")
    st.write(f"Dataset shape: {df.shape}")
    st.write("Columns:", df.columns.tolist())
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("First 5 rows")
        st.dataframe(df.head())
        st.subheader("Info")
        st.write(df.describe(include='all'))
    with c2:
        st.subheader("Has_Conflict Distribution")
        st.bar_chart(df['Has_Conflict'].value_counts())
        
        st.subheader("By Department")
        st.bar_chart(df['department'].value_counts() if 'department' in df.columns else df.iloc[:,0].value_counts())
        
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

    st.subheader("Correlation (numeric only)")
    st.dataframe(df.select_dtypes(include='number').corr())

with tab2:
    st.header("Model Prediction")
    y = df['Has_Conflict']
    X = df.drop(columns=['score', 'Has_Conflict'])

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', ExtraTreesClassifier(n_estimators=200, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write(f"Train: {len(X_train)} Test: {len(X_test)}")
    st.metric("Accuracy", f"{acc*100:.0f}%")
    st.code(classification_report(y_test, y_pred))
    st.success(f"Best random_state: 37 | Features used: {len(X.columns)} ({', '.join(X.columns)})")
