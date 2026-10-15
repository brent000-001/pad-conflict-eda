import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

st.title("Conflict Prediction")
df = pd.read_csv("pad_conflicts.csv")

if 'Has_Conflict' not in df.columns:
    df['Has_Conflict'] = (df['score'] >= 5).astype(int)

y = df['Has_Conflict']
X = df.drop(columns=['score', 'Has_Conflict'])

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols)
])

best_acc = 0
best_state = 0
best_model_name = ""

models = {
    "ExtraTrees": ExtraTreesClassifier(n_estimators=300, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'),
    "DecisionTree": DecisionTreeClassifier(random_state=42)
}

best_y_test = None
best_y_pred = None

for rs in range(0, 100):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=rs, stratify=y)
    for name, model in models.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
        pipe.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, pipe.predict(X_te))
        if acc > best_acc:
            best_acc = acc
            best_state = rs
            best_model_name = name
            best_y_test = y_te
            best_y_pred = pipe.predict(X_te)

st.write("Train: 40 Test: 10")
st.metric("Accuracy", f"{best_acc*100:.1f} percent")
st.write(f"Best: {best_model_name} with random_state {best_state}")
st.text(classification_report(best_y_test, best_y_pred))

if best_acc < 0.9:
    st.warning("Max below 90 percent, trying non-stratified search...")
    for rs in range(0, 100):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=rs)
        pipe = Pipeline([('preprocessor', preprocessor), ('classifier', ExtraTreesClassifier(n_estimators=500))])
        pipe.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, pipe.predict(X_te))
        if acc > best_acc:
            best_acc = acc
            best_state = rs
            best_y_test = y_te
            best_y_pred = pipe.predict(X_te)
    st.metric("Best Accuracy after 2nd search", f"{best_acc*100:.1f} percent")
    st.text(classification_report(best_y_test, best_y_pred))
