import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

df = pd.read_csv("/tmp/airline/data/cleaned/airline_delay_cleaned.csv")

# Target: same definition as the original study (delay_rate >= 0.25 -> "high delay" period)
df["high_delay"] = (df["delay_rate"] >= 0.25).astype(int)

# IMPORTANT: original model used total_delay_minutes / carrier_ct / weather_ct / nas_ct /
# late_aircraft_ct as predictors. Those columns are literally the arithmetic components that
# delay_rate/arr_del15 is built from, so the model was 99% just re-deriving the label from
# itself (hence the AUC=1.00 / 99.38% accuracy in the original report). Removing all of those
# and using only information that is genuinely independent of the current period's outcome:
#   - carrier identity, airport identity (encoded), month/season, flight volume (scale),
#     and prior-period (lagged) delay rate for that carrier-airport pair
df = df.sort_values(["carrier","airport","year","month"])
df["prior_delay_rate"] = df.groupby(["carrier","airport"])["delay_rate"].shift(1)
df["prior_cancellation_rate"] = df.groupby(["carrier","airport"])["cancellation_rate"].shift(1)

model_df = df.dropna(subset=["prior_delay_rate","prior_cancellation_rate"]).copy()

top_carriers = model_df["carrier"].value_counts().head(15).index
model_df["carrier_grp"] = np.where(model_df["carrier"].isin(top_carriers), model_df["carrier"], "OTHER")

X = pd.get_dummies(model_df[["carrier_grp","month"]].astype(str), drop_first=True)
X["arr_flights"] = model_df["arr_flights"].values
X["prior_delay_rate"] = model_df["prior_delay_rate"].values
X["prior_cancellation_rate"] = model_df["prior_cancellation_rate"].values
y = model_df["high_delay"].values

print("Positive class rate:", y.mean().round(3))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
num_cols = ["arr_flights","prior_delay_rate","prior_cancellation_rate"]
X_train_s, X_test_s = X_train.copy(), X_test.copy()
X_train_s[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_s[num_cols] = scaler.transform(X_test[num_cols])

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X_train_s, y_train)

y_pred = clf.predict(X_test_s)
y_prob = clf.predict_proba(X_test_s)[:,1]

print(classification_report(y_test, y_pred, digits=3))
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print("AUC:", round(roc_auc,3))

fig, axes = plt.subplots(1,2, figsize=(12,5))
import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Low Delay","High Delay"], yticklabels=["Low Delay","High Delay"])
axes[0].set_title("Confusion Matrix - Logistic Regression (leakage-free)")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="#2980b9")
axes[1].plot([0,1],[0,1],"--", color="gray")
axes[1].set_title("ROC Curve - Logistic Regression (leakage-free)")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].legend()
plt.tight_layout()
plt.savefig("/tmp/airline/visualizations/10_logreg_results_corrected.png")
plt.close()
print("Saved 10_logreg_results_corrected.png")
