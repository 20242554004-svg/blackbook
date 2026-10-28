
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# 1. Project folder and dataset
folder = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(
    folder, "PCOS_data_without_infertility.xlsx"
)

if not os.path.exists(file_path):
    raise FileNotFoundError(
        "Dataset not found. Check the Excel filename in: " + folder
    )

df = pd.read_excel(file_path, sheet_name="Full_new")
df.columns = df.columns.str.strip()
df = df.dropna(axis=1, how="all")
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

print("PCOS Dataset Loaded Successfully")
print("Dataset shape:", df.shape)

# 2. Select seven features
features = [
    "Age (yrs)",
    "BMI",
    "Cycle(R/I)",
    "Cycle length(days)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)"
]
target = "PCOS (Y/N)"

X = df[features].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[target], errors="coerce")

valid = y.isin([0, 1])
X = X.loc[valid]
y = y.loc[valid].astype(int)

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Total Records:", len(X))
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# 4. Define models
models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
    "Decision Tree": make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(max_depth=3, random_state=42)
    ),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=100, max_depth=4, random_state=42
        )
    )
}

# 5. Train and evaluate
results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, pred) * 100
    predictions[name] = pred

    print("\nModel:", name)
    print("Test Accuracy:", round(results[name], 2), "%")

# 6. Create six-chart dashboard
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("PCOS Detection - ML Dashboard", fontsize=15)

counts = y.value_counts().reindex([0, 1], fill_value=0)
axes[0, 0].bar(["No PCOS", "PCOS"], counts.values)
axes[0, 0].set_title("PCOS Distribution")
axes[0, 0].set_ylabel("Records")

age = pd.to_numeric(df["Age (yrs)"], errors="coerce")
axes[0, 1].hist(age.dropna(), bins=10)
axes[0, 1].set_title("Age Distribution")
axes[0, 1].set_xlabel("Age")

axes[0, 2].bar(list(results.keys()), list(results.values()))
axes[0, 2].set_title("Model Accuracy Comparison")
axes[0, 2].set_ylabel("Accuracy (%)")
axes[0, 2].set_ylim(0, 100)
axes[0, 2].tick_params(axis="x", rotation=20)

cycle = pd.to_numeric(df["Cycle(R/I)"], errors="coerce")
cycle_counts = cycle.value_counts().sort_index()
axes[1, 0].bar(cycle_counts.index.astype(str), cycle_counts.values)
axes[1, 0].set_title("Menstrual Cycle Type")
axes[1, 0].set_xlabel("Cycle Code")

bmi = pd.to_numeric(df["BMI"], errors="coerce")
axes[1, 1].hist(bmi.dropna(), bins=10)
axes[1, 1].set_title("BMI Distribution")
axes[1, 1].set_xlabel("BMI")

best_name = max(results, key=results.get)
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions[best_name],
    labels=[0, 1],
    display_labels=["No PCOS", "PCOS"],
    ax=axes[1, 2],
    colorbar=False
)
axes[1, 2].set_title("Best Model Confusion Matrix")

plt.tight_layout(rect=[0, 0, 1, 0.94])
dashboard_path = os.path.join(
    folder, "PCOS_Infographic_Dashboard.png"
)
plt.savefig(dashboard_path, dpi=200)
plt.show()

# 7. Save accuracy results
results_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy (%)": [round(v, 2) for v in results.values()]
})
results_df.to_csv(
    os.path.join(folder, "accuracy_results.csv"), index=False
)

print("\nBest Model:", best_name)
print("Best Test Accuracy:", round(results[best_name], 2), "%")
print("Dashboard saved:", dashboard_path)

# 8. Save trained model for Flask
model_path = os.path.join(folder, "pcos_model.joblib")
joblib.dump({
    "model": models[best_name],
    "features": features,
    "model_name": best_name
}, model_path)

print("MODEL SAVED SUCCESSFULLY AT:")
print(model_path)
