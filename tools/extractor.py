import torch
import torch.nn as nn
import timesformer.utils.logging as logging
from tqdm import tqdm
import timesformer.utils.distributed as du
from timesformer.datasets import loader
from timesformer.models import build_model
import timesformer.utils.checkpoint as cu
import numpy as np

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_extraction(
    test_loader, model, save_embedding=True, save_path="timesformer_embeddings.pt"
):
    model.eval()

    if hasattr(model, "module"):
        model.module.model.head = nn.Identity()
    else:
        model.model.head = nn.Identity()

    all_embeddings, all_labels, all_video_idxs = [], [], []
    logger.info("Starting embedding extraction...")

    for inputs, labels, video_idx, meta in tqdm(test_loader):
        if isinstance(inputs, list):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

        embeddings = model(inputs)

        all_embeddings.append(embeddings.cpu().detach())
        all_labels.append(labels.cpu().detach())
        all_video_idxs.append(video_idx.cpu().detach())

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
        }


def extract_features(cfg, save_embedding=True, save_path=None, split="train"):
    if save_embedding and save_path is None:
        raise ValueError("save_path must be provided when save_embedding=True")
    """Initializes model/data and triggers extraction."""
    du.init_distributed_training(cfg)
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

    loader_ = loader.construct_loader(cfg, split)  # ← use the split param
    return perform_extraction(loader_, model, save_embedding, save_path)
