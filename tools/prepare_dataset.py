import random
import csv
from pathlib import Path


def split_data(data, train_ratio=0.8, val_ratio=0.1):
    """Helper function to split a list into train, val, and test sets."""
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return data[:train_end], data[train_end:val_end], data[val_end:]


def create_video_splits(real_dir, fake_dir, output_dir=".", samples=None):
    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all mp4 files
    real_vids = list(Path(real_dir).glob("*.mp4"))
    fake_vids = list(Path(fake_dir).glob("*.mp4"))

    print(f"Found {len(real_vids)} Real videos and {len(fake_vids)} Fake videos.")

    # Determine the maximum possible balanced size (size of the minority class)
    max_possible_samples = min(len(real_vids), len(fake_vids))

    if samples is not None and samples < max_possible_samples:
        num_samples = samples
    else:
        num_samples = max_possible_samples

    print(
        f"Balancing classes: Randomly selecting {num_samples} videos from EACH class..."
    )

    # Randomly sample to guarantee a balanced 1:1 dataset
    real_sampled = random.sample(real_vids, num_samples)
    fake_sampled = random.sample(fake_vids, num_samples)

    # Resolve absolute paths and assign labels (0=Real, 1=Fake)
    real_data = [(str(vid.resolve()), 0) for vid in real_sampled]
    fake_data = [(str(vid.resolve()), 1) for vid in fake_sampled]

    # Shuffle each class individually before splitting
    random.shuffle(real_data)
    random.shuffle(fake_data)

    # Stratified split (80-10-10) for each class separately
    # This guarantees exactly 50% Real and 50% Fake in train, val, and test
    real_train, real_val, real_test = split_data(
        real_data, train_ratio=0.8, val_ratio=0.1
    )
    fake_train, fake_val, fake_test = split_data(
        fake_data, train_ratio=0.8, val_ratio=0.1
    )

    # Combine the class splits
    train_data = real_train + fake_train
    val_data = real_val + fake_val
    test_data = real_test + fake_test

    # Shuffle the combined sets so Real and Fake are mixed during training/testing
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # Helper function to save lists to text files
    def save_csv(data_list, filename):
        filepath = Path(output_dir) / filename
        with open(filepath, "w", newline="") as f:
            # Standard Kinetics dataloaders expect space-separated values
            writer = csv.writer(f, delimiter=",")
            for row in data_list:
                writer.writerow(row)
        print(f"Saved {len(data_list)} video paths to {filepath}")

    # Write the files
    save_csv(train_data, "train.csv")
    save_csv(val_data, "val.csv")
    save_csv(test_data, "test.csv")


if __name__ == "__main__":
    # Directory paths
    real_folder = "/media/wizav/Data/data/celeb/Celeb-real"
    fake_folder = "/media/wizav/Data/data/celeb/Celeb-synthesis"
    output_directory = "/media/wizav/Data/data/timesformer_smell_test/data"

    create_video_splits(real_folder, fake_folder, output_directory, samples=None)
