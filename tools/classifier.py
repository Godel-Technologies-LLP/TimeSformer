import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def train_classifier(train_data_path, val_data_path, save_model_path="lr_model.pkl"):
    print(f"Loading train data from {train_data_path}...")
    train_data = torch.load(train_data_path)
    X_train = train_data["embeddings"].numpy()
    y_train = train_data["labels"].numpy()

    print(f"Loading val data from {val_data_path}...")
    val_data = torch.load(val_data_path)
    X_val = val_data["embeddings"].numpy()
    y_val = val_data["labels"].numpy()

    clf = LogisticRegression(max_iter=1000, random_state=42)
    print("Training Logistic Regression...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("\n=== Validation Results ===")
    print(f"Accuracy: {accuracy_score(y_val, y_pred) * 100:.2f}%\n")
    print(classification_report(y_val, y_pred))

    joblib.dump(clf, save_model_path)
    print(f"Model saved to {save_model_path}")


def test_classifier(X_test, y_test, model_path="lr_model.pkl"):
    print(f"Loading classifier weights from {model_path}...")
    clf = joblib.load(model_path)

    print("Running inference...")
    y_pred = clf.predict(X_test)

    # If ground truth labels are provided during inference, print metrics
    if y_test is not None:
        print("\n=== Inference Evaluation ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
        print(classification_report(y_test, y_pred))

    return y_pred
