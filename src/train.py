"""
Train the Random Forest model for Phishing URL Detection.
Uses the exact configuration from notebook/01_dataset_analysis.ipynb:
- 18 URL features
- Stratified 80/20 train/test split (random_state=42)
- RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
- Saves model artifact to models/random_forest_model.joblib
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

URL_FEATURES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS"
]

def train_and_save_model(data_path="data/PhiUSIIL_Phishing_URL_Dataset.csv", model_dir="models"):
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    # Remove duplicates by URL as done in notebook
    df_clean = df.drop_duplicates(subset="URL").copy()
    print(f"After removing duplicates: {df_clean.shape}")

    X = df_clean[URL_FEATURES]
    y = df_clean["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print("Training RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)...")

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob_phishing = rf.predict_proba(X_test)[:, 0]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=0)
    rec = recall_score(y_test, y_pred, pos_label=0)
    f1 = f1_score(y_test, y_pred, pos_label=0)
    roc = roc_auc_score((y_test == 0).astype(int), y_prob_phishing)

    print("\nModel Evaluation on Test Set:")
    print(f"  Accuracy : {acc:.5f}")
    print(f"  Precision: {prec:.5f}")
    print(f"  Recall   : {rec:.5f}")
    print(f"  F1 Score : {f1:.5f}")
    print(f"  ROC-AUC  : {roc:.5f}")

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "random_forest_model.joblib")
    
    # Save model and metadata
    artifact = {
        "model": rf,
        "feature_names": URL_FEATURES,
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(roc)
        }
    }
    joblib.dump(artifact, model_path, compress=3)
    print(f"\nModel artifact saved to {model_path} (compressed)")
    return artifact

if __name__ == "__main__":
    train_and_save_model()
