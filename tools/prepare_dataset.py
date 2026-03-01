import random
import csv
from pathlib import Path


def split_data(data, train_ratio=0.8, val_ratio=0.1):
    """Helper function to split a list into train, val, and test sets."""
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return data[:train_end], data[train_end:val_end], data[val_end:]


def create_video_splits(real_dir, fake_dir, output_dir=".", samples=500):
    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all mp4 files
    real_vids = list(Path(real_dir).glob("*.mp4"))
    fake_vids = list(Path(fake_dir).glob("*.mp4"))

    # Check available videos and sample randomly
    if len(real_vids) < samples:
        print(f"Warning: Only {len(real_vids)} real videos found. Using all.")
        real_sampled = real_vids
    else:
        real_sampled = random.sample(real_vids, samples)

    if len(fake_vids) < samples:
        print(f"Warning: Only {len(fake_vids)} fake videos found. Using all.")
        fake_sampled = fake_vids
    else:
        fake_sampled = random.sample(fake_vids, samples)

    # Resolve absolute paths and assign labels (0=Real, 1=Fake)
    real_data = [(str(vid.resolve()), 0) for vid in real_sampled]
    fake_data = [(str(vid.resolve()), 1) for vid in fake_sampled]

    # Shuffle each class before splitting
    random.shuffle(real_data)
    random.shuffle(fake_data)

    # Stratified split (80-10-10) for each class separately
    # to guarantee perfectly balanced datasets
    real_train, real_val, real_test = split_data(real_data)
    fake_train, fake_val, fake_test = split_data(fake_data)

    # Combine the class splits
    train_data = real_train + fake_train
    val_data = real_val + fake_val
    test_data = real_test + fake_test

    # Shuffle the combined sets so real and fake are mixed during training
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # Helper function to save lists to CSV files
    def save_csv(data, filename):
        filepath = Path(output_dir) / filename
        with open(filepath, "w", newline="") as f:
            # Using a standard comma delimiter.
            # Change delimiter=',' to delimiter=' ' if your TimeSformer
            # dataloader specifically expects space-separated columns.
            writer = csv.writer(f, delimiter=",")
            for row in data:
                writer.writerow(row)
        print(f"Saved {len(data)} video paths to {filepath}")

    # Write the files
    save_csv(train_data, "train.csv")
    save_csv(val_data, "val.csv")
    save_csv(test_data, "test.csv")


if __name__ == "__main__":
    # Directory paths
    real_folder = "/media/wizav/Data/data/celeb/Celeb-real"
    fake_folder = "/media/wizav/Data/data/celeb/Celeb-synthesis"
    output_directory = "/media/wizav/Data/data/timesformer_smell_test/data"

    create_video_splits(real_folder, fake_folder, output_directory, samples=500)
