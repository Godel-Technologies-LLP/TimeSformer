import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class VideoDataset(Dataset):
    def __init__(self, npy_path):
        data = np.load(npy_path, allow_pickle=True).item()
        self.frames = data["frames"]
        self.metadata = data["metadata"]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        video = self.frames[idx]

        # Convert to PyTorch float tensor and normalize to [0, 1]
        tensor_video = torch.from_numpy(video).float() / 255.0

        # Original shape: (T, H, W, C) -> PyTorch Video format: (C, T, H, W)
        tensor_video = tensor_video.permute(3, 0, 1, 2)
        label = torch.tensor(meta["label"], dtype=torch.long)

        return tensor_video, label, meta


def verify_dataset(npy_path="video_dataset.npy"):
    print("--- Running Verification ---")
    dataset = VideoDataset(npy_path)
    print(f"Total samples correctly loaded into Dataset: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch_videos, batch_labels, batch_meta in loader:
        print(f"\nBatch Videos Tensor Shape: {batch_videos.shape}")
        print(
            "Expected Shape: (Batch_Size=4, Channels=3, Frames=8, Height=224, Width=224)"
        )
        print(f"Batch Labels: {batch_labels}")

        print("\nMetadata Check (First item in batch):")
        print(f"  Video Name: {batch_meta['video_name'][0]}")
        print(f"  Label (0=Real, 1=Fake): {batch_meta['label'][0].item()}")
        print(f"  Random Start Frame: {batch_meta['start_frame_index'][0].item()}")
        break


if __name__ == "__main__":
    # Example usage:
    verify_dataset(
        npy_path="/media/wizav/Data/data/timesformer_smell_test/data/processed/res_448/video_dataset.npy"
    )
