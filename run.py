import torch
import numpy as np
import torch.nn as nn
from timesformer.models.vit import TimeSformer


def extract_and_save_embeddings(dataset_path, output_path, batch_size=2):
    # 1. Load the preprocessed dataset and metadata
    print(f"Loading dataset from {dataset_path}...")
    dataset_dict = np.load(dataset_path, allow_pickle=True).item()

    frames_np = dataset_dict["frames"]
    metadata_list = dataset_dict["metadata"]

    # 2. Convert to PyTorch Tensor, permute to B C T H W, and normalize
    video_tensor = torch.from_numpy(frames_np)
    video_tensor = video_tensor.permute(0, 4, 1, 2, 3).float() / 255.0

    # 3. Initialize the TimeSformer model
    model = TimeSformer(
        img_size=224,
        num_classes=400,
        num_frames=16,
        attention_type="divided_space_time",
    )

    # Replace the classification head with Identity to extract raw embeddings
    model.model.head = nn.Identity()

    # Move model to GPU if available and set to evaluation mode
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # 4. Extract embeddings in batches
    print(f"Extracting embeddings on {device}...")
    all_embeddings = []

    with torch.no_grad():  # Disable gradient calculation to save memory
        for i in range(0, len(video_tensor), batch_size):
            batch_frames = video_tensor[i : i + batch_size].to(device)

            # Forward pass
            embeddings = model(batch_frames)

            # Move back to CPU and convert to numpy array
            all_embeddings.append(embeddings.cpu().numpy())

    # Concatenate all batch results into a single array
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    # 5. Combine metadata with embeddings
    output_results = []
    for i, meta in enumerate(metadata_list):
        output_results.append(
            {
                "video_name": meta["video_name"],
                "label": meta["label"],
                "start_frame_index": meta["start_frame_index"],
                "embedding": all_embeddings[i],
            }
        )

    # 6. Save the final results to disk
    np.save(output_path, output_results)
    print(f"Successfully saved {len(output_results)} embeddings to {output_path}")


if __name__ == "__main__":
    DATASET_PATH = "/media/wizav/Data/data/timesformer_smell_test/data/processed/frames_16/res_224/video_dataset.npy"
    OUTPUT_PATH = "/media/wizav/Data/data/timesformer_smell_test/embeddings/beta_resolution/default_224.npy"

    extract_and_save_embeddings(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PATH,
        batch_size=1,  # Adjust based on your GPU memory
    )
