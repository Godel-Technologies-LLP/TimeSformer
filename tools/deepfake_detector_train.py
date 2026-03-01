from timesformer.utils.parser import load_config, parse_args
from extractor import extract_features
from classifier import train_classifier

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)

    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True

    emb_dir = cfg.EMBEDDINGS.DIR
    train_path = f"{emb_dir}/{cfg.EMBEDDINGS.TRAIN_FILE}"
    val_path = f"{emb_dir}/{cfg.EMBEDDINGS.VAL_FILE}"
    model_path = cfg.CLASSIFIER.MODEL_PATH

    print("--- STEP 1a: Extracting Train Embeddings ---")
    extract_features(cfg, save_embedding=True, save_path=train_path, split="train")

    print("--- STEP 1b: Extracting Val Embeddings ---")
    extract_features(cfg, save_embedding=True, save_path=val_path, split="val")

    print("\n--- STEP 2: Training Classifier ---")
    train_classifier(
        train_data_path=train_path, val_data_path=val_path, save_model_path=model_path
    )
