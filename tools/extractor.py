import csv
import time
import torch
import torch.nn as nn
import timesformer.utils.logging as logging
from tqdm import tqdm
import timesformer.utils.distributed as du
from timesformer.datasets import loader
from timesformer.models import build_model
import timesformer.utils.checkpoint as cu
import numpy as np
from pathlib import Path

logger = logging.get_logger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def build_inference_csv(input_dir, output_dir):
    """
    Scans input_dir for video files. Supported files are written to
    <output_dir>/inference_input.csv with a dummy label of -1.
    Unsupported files are logged to <output_dir>/unsupported_files.txt.
    Returns the path to the generated CSV.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    supported, unsupported = [], []
    for f in sorted(input_dir.iterdir()):
        if f.is_file():
            if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                supported.append(f.resolve())
            else:
                unsupported.append(f)

    if unsupported:
        log_path = output_dir / "unsupported_files.txt"
        print(f"\n[WARNING] {len(unsupported)} unsupported file(s) in {input_dir}:")
        with open(log_path, "w") as log_f:
            for f in unsupported:
                print(f"  [UNSUPPORTED] {f}")
                log_f.write(str(f) + "\n")
        print(f"Unsupported files logged to: {log_path}\n")

    if not supported:
        raise RuntimeError(f"No supported video files found in {input_dir}")

    csv_path = output_dir / "test.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for vid in supported:
            writer.writerow([str(vid), -1])  # -1 = unknown label at inference time

    print(f"Built inference CSV with {len(supported)} file(s): {csv_path}")
    return csv_path


@torch.no_grad()
def perform_extraction(
    test_loader, model, save_embedding=True, save_path="timesformer_embeddings.pt"
):
    model.eval()

    if hasattr(model, "module"):
        model.module.model.head = nn.Identity()
    else:
        model.model.head = nn.Identity()

    all_embeddings, all_labels, all_video_idxs, all_processing_times = [], [], [], []
    logger.info("Starting embedding extraction...")

    for inputs, labels, video_idx, meta in tqdm(test_loader):
        t_start = time.perf_counter()

        if isinstance(inputs, list):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        embeddings = model(inputs)
        torch.cuda.synchronize()
        t_elapsed = time.perf_counter() - t_start

        batch_size = embeddings.shape[0]

        all_embeddings.append(embeddings.cpu().detach())
        all_labels.append(labels.cpu().detach())
        all_video_idxs.append(video_idx.cpu().detach())
        all_processing_times.extend([t_elapsed / batch_size] * batch_size)

    if not all_embeddings:
        raise RuntimeError(
            "No embeddings were extracted. Check that your input folder "
            "contains valid, supported video files."
        )

    final_embeddings = torch.cat(all_embeddings, dim=0)
    final_labels = torch.cat(all_labels, dim=0)
    final_video_idxs = torch.cat(all_video_idxs, dim=0)

    # TRAIN MODE: Save to disk
    if save_embedding:
        torch.save(
            {
                "embeddings": final_embeddings,
                "labels": final_labels,
                "video_idx": final_video_idxs,
            },
            save_path,
        )
        logger.info(f"Saved {final_embeddings.shape[0]} embeddings to {save_path}")
        return save_path

    # INFERENCE MODE: Return as numpy arrays for the classifier
    else:
        logger.info("Returning embeddings in memory for inference.")
        return {
            "embeddings": final_embeddings.numpy(),
            "labels": final_labels.numpy(),
            "video_idx": final_video_idxs.numpy(),
            "processing_times": np.array(all_processing_times),
        }


def extract_features(cfg, save_embedding=True, save_path=None, split="train"):
    if save_embedding and save_path is None:
        raise ValueError("save_path must be provided when save_embedding=True")
    """Initializes model/data and triggers extraction."""
    du.init_distributed_training(cfg)

    inference_csv = None
    if not save_embedding:
        inference_csv = build_inference_csv(cfg.INFERENCE.INPUT_DIR, cfg.OUTPUT_DIR)
        cfg.DATA.PATH_TO_DATA_DIR = str(Path(inference_csv).parent)

    model = build_model(cfg)

    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        cu.load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=False,
        )

    loader_ = loader.construct_loader(cfg, split)
    result = perform_extraction(loader_, model, save_embedding, save_path)

    if not save_embedding and isinstance(result, dict):
        # Build video_idx → filename map directly from the CSV we generated.
        # In TimeSformer's Kinetics-style loader, video_idx is always the row
        # index of the video in the CSV — this is the only reliable mapping.
        idx_to_filename = {}
        with open(inference_csv, "r") as f:
            for i, row in enumerate(csv.reader(f)):
                if row:
                    idx_to_filename[i] = Path(row[0].strip()).name
        result["idx_to_filename"] = idx_to_filename

    return result
