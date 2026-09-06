import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest

st.title("PAD Conflict Analysis")

df = pd.read_csv("pad_conflicts.csv")

st.write("Columns in CSV:", df.columns.tolist())
st.dataframe(df.head())

