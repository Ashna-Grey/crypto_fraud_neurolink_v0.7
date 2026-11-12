import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = "C:/Users/ASHNA/Desktop/Ashna/crypto-fraud-intelligence/github_files/Crypto-fraud-intelligence-main/01_Dataset/processed/ethereum_fraud_25_features.csv"

MODEL_DIR = "C:/Users/ASHNA/Desktop/Ashna/crypto-fraud-intelligence/github_files/Crypto-fraud-intelligence-main/03_Random_Forest/artifacts"

RANDOM_STATE = 42
TRAIN_SIZE = 0.70
CALIBRATION_SIZE = 0.10
CONFORMAL_SIZE = 0.10
TEST_SIZE = 0.10

N_ESTIMATORS = 1000
N_FOLDS = 5

# Feature counts evaluated during training-only CV
FEATURE_COUNTS = [5, 10, 15, 20, 25]

# Final number selected by the original experiment
BEST_N_FEATURES = 20


# ============================================================
# 2. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("=" * 80)
print("LOADING DATASET")
print("=" * 80)

df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)

if "FLAG" not in df.columns:
    raise ValueError("Target column 'FLAG' not found in dataset.")

print("\nFLAG distribution:")
print(df["FLAG"].value_counts())


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["FLAG"])
y = df["FLAG"]

print("\nNumber of features:", X.shape[1])
print("Number of samples :", X.shape[0])

print("\nTotal NaN values:", X.isna().sum().sum())


# ============================================================
# 5. TRAIN / CALIBRATION / CONFORMAL / TEST SPLIT
#
# 70% TRAIN
# 10% CALIBRATION
# 10% CONFORMAL
# 10% FINAL TEST
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE
)

X_cal, X_temp2, y_cal, y_temp2 = train_test_split(
    X_temp,
    y_temp,
    test_size=2/3,
    stratify=y_temp,
    random_state=RANDOM_STATE
)

X_conf, X_test, y_conf, y_test = train_test_split(
    X_temp2,
    y_temp2,
    test_size=0.50,
    stratify=y_temp2,
    random_state=RANDOM_STATE
)

print("\n" + "=" * 80)
print("TRAIN / CALIBRATION / CONFORMAL / TEST SPLIT")
print("=" * 80)

print("Training set   :", X_train.shape)
print("Calibration    :", X_cal.shape)
print("Conformal set  :", X_conf.shape)
print("Final test set :", X_test.shape)


print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTest class distribution:")
print(y_test.value_counts())


# ============================================================
# 6. TRAINING-ONLY 5-FOLD CROSS-VALIDATION
#
# For every fold:
#
#   CV train
#       ↓
#   fit imputer
#       ↓
#   transform CV train + validation
#       ↓
#   rank features using CV train only
#       ↓
#   select top N
#       ↓
#   train RF
#       ↓
#   evaluate validation fold
#
# The final test set is NEVER used here.
# ============================================================

print("\n" + "=" * 80)
print("TRAINING-ONLY 5-FOLD FEATURE COUNT SELECTION")
print("=" * 80)

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_results = []


for n_features in FEATURE_COUNTS:

    print("\n" + "=" * 80)
    print(f"EVALUATING TOP {n_features} FEATURES")
    print("=" * 80)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X_train, y_train),
        start=1
    ):

        print(f"\nFold {fold}/{N_FOLDS}")

        # ----------------------------------------------------
        # CV TRAIN / VALIDATION SPLIT
        # ----------------------------------------------------

        X_tr = X_train.iloc[train_idx].copy()
        X_val = X_train.iloc[val_idx].copy()

        y_tr = y_train.iloc[train_idx].copy()
        y_val = y_train.iloc[val_idx].copy()

        # ----------------------------------------------------
        # IMPUTATION
        #
        # Fit ONLY on CV training fold.
        # ----------------------------------------------------

        imputer = SimpleImputer(strategy="median")

        X_tr_imp = pd.DataFrame(
            imputer.fit_transform(X_tr),
            columns=X_tr.columns,
            index=X_tr.index
        )

        X_val_imp = pd.DataFrame(
            imputer.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )

        # ----------------------------------------------------
        # FEATURE RANKING
        #
        # Ranking model sees ONLY CV training data.
        # ----------------------------------------------------

        ranking_model = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        ranking_model.fit(
            X_tr_imp,
            y_tr
        )

        feature_ranking = pd.DataFrame({
            "Feature": X_tr_imp.columns,
            "Importance": ranking_model.feature_importances_
        }).sort_values(
            by="Importance",
            ascending=False
        ).reset_index(drop=True)

        # ----------------------------------------------------
        # SELECT TOP N FEATURES
        # ----------------------------------------------------

        selected_features = (
            feature_ranking
            .head(n_features)["Feature"]
            .tolist()
        )

        # ----------------------------------------------------
        # TRAIN RANDOM FOREST
        # ----------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(
            X_tr_imp[selected_features],
            y_tr
        )

        # ----------------------------------------------------
        # VALIDATION PREDICTION
        # ----------------------------------------------------

        y_pred = model.predict(
            X_val_imp[selected_features]
        )

        y_prob = model.predict_proba(
            X_val_imp[selected_features]
        )[:, 1]

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_val,
            y_pred
        )

        precision = precision_score(
            y_val,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_val,
            y_prob
        )

        pr_auc = average_precision_score(
            y_val,
            y_prob
        )

        fold_results.append({
            "fold": fold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        })

        print(
            f"F1={f1:.6f} | "
            f"ROC-AUC={roc_auc:.6f} | "
            f"PR-AUC={pr_auc:.6f}"
        )

    # --------------------------------------------------------
    # AVERAGE FOLD RESULTS
    # --------------------------------------------------------

    fold_df = pd.DataFrame(fold_results)

    cv_results.append({
        "Features": n_features,

        "Accuracy Mean":
            fold_df["accuracy"].mean(),

        "Accuracy Std":
            fold_df["accuracy"].std(),

        "Precision Mean":
            fold_df["precision"].mean(),

        "Precision Std":
            fold_df["precision"].std(),

        "Recall Mean":
            fold_df["recall"].mean(),

        "Recall Std":
            fold_df["recall"].std(),

        "F1 Mean":
            fold_df["f1"].mean(),

        "F1 Std":
            fold_df["f1"].std(),

        "ROC-AUC Mean":
            fold_df["roc_auc"].mean(),

        "ROC-AUC Std":
            fold_df["roc_auc"].std(),

        "PR-AUC Mean":
            fold_df["pr_auc"].mean(),

        "PR-AUC Std":
            fold_df["pr_auc"].std()
    })


# ============================================================
# 7. CV COMPARISON
# ============================================================

cv_results_df = pd.DataFrame(cv_results)

print("\n" + "=" * 100)
print("LEAKAGE-FREE TRAINING-ONLY FEATURE COUNT COMPARISON")
print("=" * 100)

print(
    cv_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# 8. SELECT BEST FEATURE COUNT
#
# This is based ONLY on training-data CV.
# ============================================================

best_row = cv_results_df.loc[
    cv_results_df["F1 Mean"].idxmax()
]

best_n = int(best_row["Features"])

print("\n" + "=" * 80)
print("OPTIMAL FEATURE COUNT")
print("=" * 80)

print("Best number of features:", best_n)
print(f"Mean CV F1       : {best_row['F1 Mean']:.6f}")
print(f"CV F1 Std        : {best_row['F1 Std']:.6f}")
print(f"Mean CV ROC-AUC  : {best_row['ROC-AUC Mean']:.6f}")
print(f"Mean CV PR-AUC   : {best_row['PR-AUC Mean']:.6f}")


# ============================================================
# 9. TRAINING-ONLY IMPUTATION FOR FEATURE RANKING
# ============================================================

ranking_imputer = SimpleImputer(
    strategy="median"
)

X_train_rank_imp = pd.DataFrame(
    ranking_imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_rank_imp = pd.DataFrame(
    ranking_imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


# ============================================================
# 10. FINAL FEATURE RANKING
#
# Ranking model is trained using the complete TRAINING set.
# Test set is still untouched.
# ============================================================

print("\n" + "=" * 80)
print("FINAL TRAINING-ONLY FEATURE RANKING")
print("=" * 80)

ranking_model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

ranking_model.fit(
    X_train_rank_imp,
    y_train
)

feature_ranking = pd.DataFrame({
    "Feature": X_train_rank_imp.columns,
    "Importance": ranking_model.feature_importances_
})

feature_ranking = feature_ranking.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

feature_ranking["Rank"] = (
    feature_ranking.index + 1
)

feature_ranking = feature_ranking[
    ["Rank", "Feature", "Importance"]
]

# ============================================================
# 11. SELECT FINAL FEATURES
#
# The original experiment selected 20 features.
#
# We verify that CV also selected 20. If not, stop rather
# than silently creating a different model.
# ============================================================

if best_n != BEST_N_FEATURES:
    raise RuntimeError(
        f"Training CV selected {best_n} features, "
        f"but expected {BEST_N_FEATURES}. "
        "Review the CV results before training the final model."
    )

final_features = (
    feature_ranking
    .head(BEST_N_FEATURES)["Feature"]
    .tolist()
)

print("\nSelected features:")

for i, feature in enumerate(
    final_features,
    start=1
):

    importance = feature_ranking.loc[
        feature_ranking["Feature"] == feature,
        "Importance"
    ].iloc[0]

    print(
        f"{i:2d}. "
        f"{feature:<60} "
        f"{importance:.6f}"
    )
# ============================================================
# 11A. FINAL 20-FEATURE IMPUTER
#
# This imputer is fitted ONLY on the final 20 features.
# It is the imputer that will be saved and used at inference.
# ============================================================

final_imputer = SimpleImputer(
    strategy="median"
)

X_train_final = pd.DataFrame(
    final_imputer.fit_transform(
        X_train[final_features]
    ),
    columns=final_features,
    index=X_train.index
)

X_cal_final = pd.DataFrame(
    final_imputer.transform(
        X_cal[final_features]
    ),
    columns=final_features,
    index=X_cal.index
)

X_conf_final = pd.DataFrame(
    final_imputer.transform(
        X_conf[final_features]
    ),
    columns=final_features,
    index=X_conf.index
)

X_test_final = pd.DataFrame(
    final_imputer.transform(
        X_test[final_features]
    ),
    columns=final_features,
    index=X_test.index
)

# ============================================================
# 12. TRAIN FINAL RANDOM FOREST
# ============================================================

print("\n" + "=" * 80)
print("TRAINING FINAL RANDOM FOREST")
print("=" * 80)

final_rf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

final_rf.fit(
    X_train_final[final_features],
    y_train
)
# ============================================================
# 12A. SAVE CALIBRATION / CONFORMAL SPLIT DATA
# ============================================================

split_data = {
    "train_indices": X_train.index.to_numpy(),
    "calibration_indices": X_cal.index.to_numpy(),
    "conformal_indices": X_conf.index.to_numpy(),
    "test_indices": X_test.index.to_numpy()
}

split_path = os.path.join(
    MODEL_DIR,
    "rf_data_split_indices.joblib"
)

joblib.dump(
    split_data,
    split_path
)

print("Saved:", split_path)


# ============================================================
# 13. FINAL TEST EVALUATION
#
# The test set is used here ONCE for final evaluation.
# ============================================================

y_test_pred = final_rf.predict(
    X_test_final[final_features]
)

y_test_prob = final_rf.predict_proba(
    X_test_final[final_features]
)[:, 1]


# ============================================================
# 14. FINAL METRICS
# ============================================================

final_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

final_precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

final_recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

final_f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)

final_roc_auc = roc_auc_score(
    y_test,
    y_test_prob
)

final_pr_auc = average_precision_score(
    y_test,
    y_test_prob
)


# ============================================================
# 15. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 80)
print("FINAL UNTOUCHED TEST-SET RESULTS")
print("=" * 80)

print(f"Features : {BEST_N_FEATURES}")
print(f"Accuracy : {final_accuracy:.6f}")
print(f"Precision: {final_precision:.6f}")
print(f"Recall   : {final_recall:.6f}")
print(f"F1       : {final_f1:.6f}")
print(f"ROC-AUC  : {final_roc_auc:.6f}")
print(f"PR-AUC   : {final_pr_auc:.6f}")


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_test_pred
)

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(cm)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)

print(
    classification_report(
        y_test,
        y_test_pred,
        digits=6
    )
)


# ============================================================
# 18. SAVE FEATURE RANKING
# ============================================================

ranking_path = os.path.join(
    MODEL_DIR,
    "rf_final_20_feature_ranking.csv"
)

feature_ranking.to_csv(
    ranking_path,
    index=False
)

print("\nSaved:", ranking_path)


# ============================================================
# 19. SAVE SELECTED FEATURES
# ============================================================

selected_df = pd.DataFrame({
    "Rank": range(1, BEST_N_FEATURES + 1),
    "Feature": final_features
})

features_path = os.path.join(
    MODEL_DIR,
    "rf_final_20_features.csv"
)

selected_df.to_csv(
    features_path,
    index=False
)

print("Saved:", features_path)


# ============================================================
# 20. SAVE CV RESULTS
# ============================================================

cv_path = os.path.join(
    MODEL_DIR,
    "rf_training_only_feature_count_cv.csv"
)

cv_results_df.to_csv(
    cv_path,
    index=False
)

print("Saved:", cv_path)


# ============================================================
# 21. SAVE RANDOM FOREST
#
# Keep ONE canonical model format for GitHub.
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "rf_ethereum_fraud_20features.joblib"
)

joblib.dump(
    final_rf,
    model_path
)

print("Saved:", model_path)


# ============================================================
# 22. SAVE IMPUTER
# ============================================================

imputer_path = os.path.join(
    MODEL_DIR,
    "rf_ethereum_fraud_20features_imputer.joblib"
)

joblib.dump(
    final_imputer,
    imputer_path
)

print("Saved:", imputer_path)


# ============================================================
# 23. SAVE MODEL CONFIGURATION
# ============================================================

model_config = {
    "model": "RandomForestClassifier",
    "task": "Ethereum fraud detection",
    "target": "FLAG",

    "n_features": BEST_N_FEATURES,
    "n_estimators": N_ESTIMATORS,

    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,

    "test_size": TEST_SIZE,

    "imputation": {
        "method": "median",
        "fit_on": "training_data_only"
    },

    "feature_selection": {
        "method": "Random Forest feature importance",
        "evaluation": "5-fold stratified cross-validation",
        "selection_metric": "F1",
        "candidate_feature_counts": FEATURE_COUNTS,
        "selected_feature_count": BEST_N_FEATURES
    }
}

config_path = os.path.join(
    MODEL_DIR,
    "model_config.json"
)

with open(
    config_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        model_config,
        f,
        indent=4
    )

print("Saved:", config_path)


# ============================================================
# 24. SAVE FEATURE SCHEMA
# ============================================================

feature_schema = {
    "target": "FLAG",
    "feature_count": len(final_features),
    "features": final_features,
    "feature_order": final_features
}

schema_path = os.path.join(
    MODEL_DIR,
    "feature_schema.json"
)

with open(
    schema_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        feature_schema,
        f,
        indent=4
    )

print("Saved:", schema_path)


# ============================================================
# 25. SAVE CLASS MAPPING
# ============================================================

class_mapping = {
    "0": "non-fraud",
    "1": "fraud"
}

class_mapping_path = os.path.join(
    MODEL_DIR,
    "class_mapping.json"
)

with open(
    class_mapping_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        class_mapping,
        f,
        indent=4
    )

print("Saved:", class_mapping_path)


# ============================================================
# 26. SAVE TRAINING METADATA
# ============================================================

metadata = {
    "model": "RandomForestClassifier",
    "task": "Ethereum fraud detection",
    "target": "FLAG",

    "selected_feature_count": BEST_N_FEATURES,
    "features": final_features,

    "n_estimators": N_ESTIMATORS,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,

    "test_size": TEST_SIZE,

    "imputation": "median",

    "feature_selection": (
        "Training-only 5-fold stratified cross-validation"
    ),

    "cv_mean_f1": float(best_row["F1 Mean"]),
    "cv_mean_roc_auc": float(best_row["ROC-AUC Mean"]),
    "cv_mean_pr_auc": float(best_row["PR-AUC Mean"]),

    "final_test_accuracy": float(final_accuracy),
    "final_test_precision": float(final_precision),
    "final_test_recall": float(final_recall),
    "final_test_f1": float(final_f1),
    "final_test_roc_auc": float(final_roc_auc),
    "final_test_pr_auc": float(final_pr_auc)
}

metadata_path = os.path.join(
    MODEL_DIR,
    "training_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )

print("Saved:", metadata_path)


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

print("\nFinal feature count:", BEST_N_FEATURES)

print("\nFinal features:")

for i, feature in enumerate(final_features, 1):
    print(f"{i:2d}. {feature}")

print("\nFinal test metrics:")
print(f"Accuracy : {final_accuracy:.6f}")
print(f"Precision: {final_precision:.6f}")
print(f"Recall   : {final_recall:.6f}")
print(f"F1       : {final_f1:.6f}")
print(f"ROC-AUC  : {final_roc_auc:.6f}")
print(f"PR-AUC   : {final_pr_auc:.6f}")

print("\nArtifacts saved to:")
print(os.path.abspath(MODEL_DIR))
