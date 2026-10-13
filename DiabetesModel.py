
#IMPORTS AND SETUP
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
)

sns.set_theme(style="whitegrid") 

SHOW_PLOTS = False                     
os.makedirs("figures", exist_ok=True)  


def save_figure(fig, filename):
    """Save a figure into figures/, optionally show it, then close it."""
    fig.savefig(f"figures/{filename}", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# %% 02. LOAD DATA AND MARK IMPOSSIBLE ZEROS AS MISSING
df = pd.read_csv("diabetes.csv")

zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
ml_df = df.copy()
ml_df[zero_cols] = ml_df[zero_cols].replace

print("Missing values going into the ML stage:")
print(ml_df.isnull().sum())



# The target has two categories (0 = no diabetes, 1 = diabetes),so we use classification
# we predict a CLASS, not a continuous number.
features = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
X = ml_df[features]      

y = ml_df["Outcome"]     

print("\nX shape:", X.shape, "| y shape:", y.shape)
print(X.dtypes)

# 80% to train, 20% to test. stratify=y keeps the same share of diabetic
# patients (about 35%) in both sets. random_state makes the split repeatable.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nTrain size:", len(X_train), "| Test size:", len(X_test))
print("Share with diabetes - train: {:.1f}%, test: {:.1f}%".format(
    y_train.mean() * 100, y_test.mean() * 100))


# Each model is a Pipeline: step 1 fills missing values with the training
# median, (step 2 scales, for logistic regression only), then the model.
# Scaling matters for logistic regression because it is sensitive to the size
# of the numbers; trees do not need it.
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ]),
    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]),
}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    print(f"Trained: {name}")


# EVALUATE ALL MODELS
rows = []
confusion_rows = []

for name, pipe in models.items():
    y_pred = pipe.predict(X_test)                 
    y_prob = pipe.predict_proba(X_test)[:, 1]      

    # 5-fold cross-validation on the TRAINING data: a steadier estimate than
    # one small test set (our test set only has ~154 patients)
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")

    rows.append({
        "Model": name,
        "Train Accuracy": pipe.score(X_train, y_train),   # to spot overfitting
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "CV ROC-AUC (mean)": cv_auc.mean(),
        "CV ROC-AUC (std)": cv_auc.std(),
    })

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    confusion_rows.append({
        "Model": name,
        "True Negatives": tn, "False Positives": fp,
        "False Negatives": fn, "True Positives": tp,
    })

results = pd.DataFrame(rows).set_index("Model").round(3)
confusion_table = pd.DataFrame(confusion_rows).set_index("Model")

print("\n=== MODEL COMPARISON (test set) ===")
print(results)
print("\n=== CONFUSION MATRIX COUNTS ===")
print(confusion_table)

results.to_csv("model_comparison.csv")

# Best model = highest ROC-AUC on the test set
best_name = results["ROC-AUC"].idxmax()
best_model = models[best_name]
print(f"\nBest model by ROC-AUC: {best_name}")


#CONFUSION MATRIX FOR THE BEST MODEL
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(
    best_model, X_test, y_test,
    display_labels=["No diabetes", "Diabetes"],
    cmap="Blues", ax=ax,
)
ax.set_title(f"Confusion matrix: {best_name}")
ax.grid(False)
save_figure(fig, "fig19_confusion_matrix.png")


#ROC CURVES FOR ALL MODELS
fig, ax = plt.subplots(figsize=(7, 6))
for name, pipe in models.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guessing")
ax.set_title("ROC curves (test set)")
ax.legend(loc="lower right")
save_figure(fig, "fig20_roc_curves.png")


#FEATURE IMPORTANCE (RANDOM FOREST)
rf_pipeline = models["Random Forest"]
rf_model = rf_pipeline.named_steps["model"]      # the forest inside the pipeline

importance = pd.DataFrame({
    "feature": features,
    "importance": rf_model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\n=== RANDOM FOREST FEATURE IMPORTANCE ===")
print(importance.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importance["feature"], importance["importance"], color="steelblue")
ax.invert_yaxis()                                # most important at the top
ax.set_xlabel("Importance")
ax.set_title("Random Forest feature importance")
save_figure(fig, "fig21_feature_importance.png")


#  PREDICT HYPOTHETICAL PATIENTS
new_patients = pd.DataFrame({
    "Pregnancies": [1, 5, 8],
    "Glucose": [100, 150, 190],
    "BloodPressure": [70, 80, 90],
    "SkinThickness": [20, 30, 35],
    "Insulin": [80, 150, 200],
    "BMI": [23, 31, 36],
    "DiabetesPedigreeFunction": [0.2, 0.6, 1.0],
    "Age": [25, 45, 60],
})

predictions = rf_pipeline.predict(new_patients[features])
probabilities = rf_pipeline.predict_proba(new_patients[features])[:, 1]

prediction_table = new_patients.copy()
prediction_table["Predicted probability of positive class"] = probabilities.round(2)
prediction_table["Model classification"] = np.where(predictions == 1, "Positive", "Negative")
prediction_table.index = ["Hypothetical 1", "Hypothetical 2", "Hypothetical 3"]

print("\n=== HYPOTHETICAL OBSERVATIONS (educational only, not a diagnosis) ===")
print(prediction_table.T)
prediction_table.to_csv("hypothetical_predictions.csv")

print("\nDone. Figures are in 'figures/'. Tables: model_comparison.csv, hypothetical_predictions.csv")

