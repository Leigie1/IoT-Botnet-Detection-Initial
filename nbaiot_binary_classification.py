"""
N-BaIoT binary classification (benign = 0, attack = 1) with Random Forest.

Layout expected (UCI flat files), e.g. in DATA_DIR:
    1.benign.csv, 1.gafgyt.combo.csv, ..., 1.mirai.udpplain.csv

Install:  pip install pandas numpy scikit-learn matplotlib seaborn
Run:      python nbaiot_binary_classification.py
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# SETTINGS (change these)
# ----------------------------------------------------------------------------
DATA_DIR = "."            # folder that contains 1.benign.csv, 1.gafgyt.*.csv ...
DEVICE = 1                # device number (1 = Danmini Doorbell)
BENIGN_ROWS = 40_000      # rows sampled from the benign file
ATTACK_ROWS_PER_FILE = 6_000  # rows sampled from EACH of the 10 attack files
CORR_THRESHOLD = 0.95     # drop one of any feature pair correlated above this
TEST_SIZE = 0.2
SEED = 42
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. PROBLEM
# ----------------------------------------------------------------------------
# Supervised binary classification: given 115 traffic statistics of an IoT
# device, predict 0 = benign traffic, 1 = botnet attack (Mirai or BASHLITE).

# ----------------------------------------------------------------------------
# 2. ALGORITHM: Random Forest
# ----------------------------------------------------------------------------
# Good for numeric tabular data, robust to correlated features, little tuning,
# resists overfitting via bagging, supports class_weight, gives feature
# importances.

# ----------------------------------------------------------------------------
# 3. LOAD AND EXPLORE
# ----------------------------------------------------------------------------
def load_device(data_dir, device):
    """Load one device's 11 files, sample rows, label from the file name."""
    frames = []
    pattern = os.path.join(data_dir, f"{device}.*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match {pattern}. Check DATA_DIR/DEVICE.")

    for path in files:
        name = os.path.basename(path)             # e.g. 1.mirai.syn.csv
        parts = name.replace(".csv", "").split(".")
        if parts[1] == "benign":
            label, attack_type, n = 0, "benign", BENIGN_ROWS
        else:
            label, attack_type, n = 1, f"{parts[1]}.{parts[2]}", ATTACK_ROWS_PER_FILE

        df = pd.read_csv(path, dtype="float32")
        df = df.sample(n=min(n, len(df)), random_state=SEED)
        df["label"] = label
        df["attack_type"] = attack_type           # kept for analysis only
        frames.append(df)
        print(f"  loaded {name:<24} -> {len(df):>6} rows (label={label})")

    return pd.concat(frames, ignore_index=True)


print("=== 3. LOAD AND EXPLORE ===")
data = load_device(DATA_DIR, DEVICE)
print("\nShape:", data.shape)
print("\nClass counts (0 = benign, 1 = attack):")
print(data["label"].value_counts())
print("\nRows per traffic type:")
print(data["attack_type"].value_counts())

feature_cols = [c for c in data.columns if c not in ("label", "attack_type")]
print(f"\nNumber of features: {len(feature_cols)}")
print("\nSummary of the first 5 features:")
print(data[feature_cols[:5]].describe().T)

# Plot: class distribution
plt.figure(figsize=(7, 4))
data["attack_type"].value_counts().plot(kind="bar")
plt.title("Rows per traffic type (after sampling)")
plt.ylabel("rows")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_class_distribution.png", dpi=120)
plt.close()

# Plot: a few feature distributions, benign vs attack
sample_feats = feature_cols[:4]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, col in zip(axes, sample_feats):
    sns.kdeplot(data=data, x=col, hue="label", common_norm=False,
                ax=ax, warn_singular=False)
    ax.set_title(col, fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_feature_distributions.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------------
# 4. PREPROCESSING
# ----------------------------------------------------------------------------
print("\n=== 4. PREPROCESSING ===")

# 4a. Cleaning
print("Missing values:", int(data[feature_cols].isnull().sum().sum()))
n_dups = int(data.duplicated(subset=feature_cols).sum())
print("Duplicate rows:", n_dups)
data = data.drop_duplicates(subset=feature_cols).reset_index(drop=True)

constant_cols = [c for c in feature_cols if data[c].nunique() <= 1]
print("Constant columns:", len(constant_cols))
feature_cols = [c for c in feature_cols if c not in constant_cols]

# 4b. Feature selection: remove highly correlated features
corr = data[feature_cols].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if (upper[c] > CORR_THRESHOLD).any()]
print(f"Dropping {len(to_drop)} features with correlation > {CORR_THRESHOLD}")
feature_cols = [c for c in feature_cols if c not in to_drop]
print("Features remaining:", len(feature_cols))

# Plot: correlation heatmap of remaining features
plt.figure(figsize=(9, 7))
sns.heatmap(data[feature_cols].corr(), cmap="coolwarm", center=0,
            xticklabels=False, yticklabels=False)
plt.title("Correlation of remaining features")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_correlation_heatmap.png", dpi=120)
plt.close()

X = data[feature_cols]
y = data["label"].astype(int)

# ----------------------------------------------------------------------------
# 5. TRAIN / TEST SPLIT (stratified)
# ----------------------------------------------------------------------------
print("\n=== 5. SPLIT ===")
X_train, X_test, y_train, y_test, type_train, type_test = train_test_split(
    X, y, data["attack_type"],
    test_size=TEST_SIZE, stratify=y, random_state=SEED)
print("Train:", X_train.shape, " Test:", X_test.shape)
print("Attack share  train: %.3f  test: %.3f" % (y_train.mean(), y_test.mean()))

# 4c. Transformation: scale (fit on TRAIN only to avoid leakage)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ----------------------------------------------------------------------------
# 6. INITIAL MODEL
# ----------------------------------------------------------------------------
print("\n=== 6. TRAIN RANDOM FOREST ===")
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    n_jobs=-1,
    random_state=SEED,
)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]

# ----------------------------------------------------------------------------
# 7. EVALUATION
# ----------------------------------------------------------------------------
print("\n=== 7. EVALUATION (random stratified split) ===")
print(classification_report(y_test, y_pred, target_names=["benign", "attack"], digits=4))
print("Confusion matrix [rows = true, cols = predicted]:")
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC: %.4f" % roc_auc_score(y_test, y_prob))
print("PR-AUC : %.4f" % average_precision_score(y_test, y_prob))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, display_labels=["benign", "attack"], cmap="Blues")
plt.title("Confusion matrix")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_confusion_matrix.png", dpi=120)
plt.close()

# Feature importance
imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 15 features by importance:")
print(imp.head(15))
plt.figure(figsize=(8, 5))
imp.head(15)[::-1].plot(kind="barh")
plt.title("Top 15 feature importances")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_feature_importance.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------------
# BONUS: stricter test, hold out entire attack types the model never saw
# ----------------------------------------------------------------------------
print("\n=== BONUS: HELD-OUT ATTACK TYPES ===")
HELD_OUT = ["mirai.udpplain", "gafgyt.junk"]   # change as you like

is_held = type_train.isin(HELD_OUT) | type_test.isin(HELD_OUT)
all_X = pd.concat([X_train, X_test])
all_y = pd.concat([y_train, y_test])
all_t = pd.concat([type_train, type_test])
held_mask = all_t.isin(HELD_OUT)

# Benign rows are split by the original train/test assignment; held-out attack
# types appear ONLY in the test set.
Xtr = pd.concat([X_train[~type_train.isin(HELD_OUT)]])
ytr = y_train[~type_train.isin(HELD_OUT)]
Xte = pd.concat([X_test[type_test == "benign"], all_X[held_mask]])
yte = pd.concat([y_test[type_test == "benign"], all_y[held_mask]])

sc2 = StandardScaler().fit(Xtr)
m2 = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                            n_jobs=-1, random_state=SEED).fit(sc2.transform(Xtr), ytr)
p2 = m2.predict(sc2.transform(Xte))
print(f"Held-out attack types: {HELD_OUT}")
print(classification_report(yte, p2, target_names=["benign", "attack"], digits=4))

print(f"\nDone. Plots saved in ./{OUT_DIR}/")
