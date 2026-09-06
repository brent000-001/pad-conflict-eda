# PAD Conflict Analysis - Yaoundé | End-to-End HR Dashboard

🚀 **Live App:** https://pad-conflict-eda-fgjg3s3appwxib4wvrpa3c.streamlit.app/

Predictive analytics dashboard for Port Authority workplace conflicts (50 cases).

## 📊 Features
- **Severity Classification:** Low (13) / Medium (25) / High (9) - Class imbalance handled
- **ML Pipeline:** StandardScaler, IsolationForest (5% contamination, 2 flagged), VIF check
- **Treatment:** class_weight='balanced' + stratified sampling
- **Corporate Export:** 1-click Excel download for HR Director (openpyxl)

## 🛠️ Tech Stack
Python, Pandas, Scikit-Learn, Statsmodels, Streamlit, Plotly, Openpyxl

## ▶️ Run Locally
pip install -r requirements.txt
streamlit run app.py

## 👤 Author
brent000-001 | Yaoundé, CM | Sept 2026
GitHub: github.com/brent000-001/pad-conflict-eda
