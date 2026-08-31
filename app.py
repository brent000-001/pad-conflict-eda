import streamlit as st
import pandas as pd

st.title("PAD Conflict Analysis - Yaounde")
st.write("EDA Virtualization Complete")

df = pd.DataFrame({
  "department": ["Finance","HR","Logistics","Operations"],
  "score": [4.37, 5.00, 6.20, 5.49]
})
st.bar_chart(df.set_index("department"))
st.success("Deployed from Chromebook Linux!")

