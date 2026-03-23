import csv
import torch
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

CLASS_NAMES = {0: "Real", 1: "Fake"}


def train_classifier(train_data_path, val_data_path, save_model_path="lr_model.pkl"):
    print(f"Loading train data from {train_data_path}...")
    train_data = torch.load(train_data_path)
    X_train = train_data["embeddings"].numpy()
    y_train = train_data["labels"].numpy()

    print(f"Loading val data from {val_data_path}...")
    val_data = torch.load(val_data_path)
    X_val = val_data["embeddings"].numpy()
    y_val = val_data["labels"].numpy()

    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    print("Training Logistic Regression...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("\n=== Validation Results ===")
    print(f"Accuracy: {accuracy_score(y_val, y_pred) * 100:.2f}%\n")
    print(classification_report(y_val, y_pred))

    joblib.dump(clf, save_model_path)
    print(f"Model saved to {save_model_path}")


def test_classifier(
    X_test,
    y_test,
    model_path="lr_model.pkl",
    video_idxs=None,
    idx_to_filename=None,
    processing_times=None,
    output_dir=None,
):
    print(f"Loading classifier weights from {model_path}...")
    clf = joblib.load(model_path)

    print("Running inference...")
    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)  # shape: (N, n_classes)

    chunk_results = []
    if video_idxs is not None and idx_to_filename is not None:
        # chunk_counter keyed on video_idx (int) — increments each time
        # the same video_idx appears, giving chunk000, chunk001, etc.
        chunk_counter = {}

        for i in range(len(video_idxs)):
            vid_idx = int(video_idxs[i])
            filename = idx_to_filename.get(vid_idx, f"unknown_{vid_idx}")

            chunk_num = chunk_counter.get(vid_idx, 0)
            chunk_counter[vid_idx] = chunk_num + 1
            chunk_name = f"{Path(filename).stem}_chunk{chunk_num:03d}"

            pred_class = int(y_pred[i])
            fake_score = float(y_scores[i][1])  # class 1 = Fake probability
            proc_time = (
                float(processing_times[i]) if processing_times is not None else "N/A"
            )

            chunk_results.append(
                {
                    "filename": filename,
                    "chunk_name": chunk_name,
                    "fake_score": round(fake_score, 4),
                    "prediction_class": CLASS_NAMES.get(pred_class, str(pred_class)),
                    "processing_time_s": (
                        round(proc_time, 4)
                        if isinstance(proc_time, float)
                        else proc_time
                    ),
                }
            )

        # Print results table
        print("\n=== Chunk-Level Results ===")
        header = f"{'Filename':<40} {'Chunk':<25} {'Fake Score':>10} {'Class':>6} {'Time(s)':>9}"
        print(header)
        print("-" * len(header))
        for r in chunk_results:
            print(
                f"{r['filename']:<40} {r['chunk_name']:<25} "
                f"{r['fake_score']:>10.4f} {r['prediction_class']:>6} "
                f"{str(r['processing_time_s']):>9}"
            )

        if output_dir:
            out_path = Path(output_dir) / "inference_results.csv"
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=chunk_results[0].keys())
                writer.writeheader()
                writer.writerows(chunk_results)
            print(f"\nResults saved to: {out_path}")

    # Aggregate metrics only when ground truth is available
    valid_mask = y_test != -1
    if valid_mask.any():
        print("\n=== Inference Evaluation (labelled samples) ===")
        print(
            f"Accuracy: {accuracy_score(y_test[valid_mask], y_pred[valid_mask]) * 100:.2f}%\n"
        )
        print(classification_report(y_test[valid_mask], y_pred[valid_mask]))

    return y_pred, chunk_results
