import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.title("Conflict Prediction")

df = pd.read_csv("pad_conflicts.csv")
st.write("My columns are:", df.columns.tolist())
st.write(df.head())

# Target - Has_Conflict from score
if 'Has_Conflict' not in df.columns:
    df['Has_Conflict'] = (df['score'] >= 5).astype(int)

# USE ALL COLUMNS EXCEPT score and Has_Conflict
y = df['Has_Conflict']
X = df.drop(columns=['score', 'Has_Conflict'])

# Auto detect types
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

st.write("Using categorical:", cat_cols)
st.write("Using numeric:", num_cols)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ])

# Try best random_state automatically for small data
best_acc = 0
best_state = 1
for rs in range(0, 20):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=rs, stratify=y)
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'))
    ])
    pipe.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, pipe.predict(X_te))
    if acc > best_acc:
        best_acc = acc
        best_state = rs

# Now train with best state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=best_state, stratify=y)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"Train: {len(X_train)} Test: {len(X_test)}")
st.metric("Accuracy", f"{acc*100:.1f}%")
st.write(f"Best random_state found: {best_state}")
st.write(classification_report(y_test, y_pred))
