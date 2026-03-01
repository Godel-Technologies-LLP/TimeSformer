from timesformer.utils.parser import load_config, parse_args
from extractor import extract_features
from classifier import test_classifier

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)

    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True

    model_path = cfg.CLASSIFIER.MODEL_PATH

    print("--- STEP 1: Extracting Embeddings to Memory ---")
    inference_data = extract_features(cfg, save_embedding=False, split="test")

    print("\n--- STEP 2: Running Inference ---")
    predictions = test_classifier(
        X_test=inference_data["embeddings"],
        y_test=inference_data["labels"],
        model_path=model_path,
    )

    print("\nInference complete. Sample predictions:", predictions[:10])
