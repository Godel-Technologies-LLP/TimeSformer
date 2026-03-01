import os
import cv2
import numpy as np
import random


def process_video_list(video_list, output_npy, n, img_size):
    # Read the text file
    with open(video_list, "r") as f:
        lines = f.read().splitlines()

    frames_array = []
    metadata_list = []

    for line in lines:
        if not line.strip():
            continue

        # rsplit handles potential spaces in the file path safely [web:13]
        vid_path, label = line.rsplit(" ", 1)
        label = int(label)
        vid_name = os.path.basename(vid_path)

        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < n:
            print(f"Skipping {vid_name}: only has {total_frames} frames.")
            cap.release()
            continue

        # Pick a random starting frame
        start_idx = random.randint(0, total_frames - n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        video_frames = []
        for _ in range(n):
            ret, frame = cap.read()
            if not ret:
                break

            # Resize to 224x224 (Standard for PyTorch vision models)
            frame = cv2.resize(frame, (img_size, img_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame)

        cap.release()

        # If we successfully read exactly n frames, add to our lists
        if len(video_frames) == n:
            frames_array.append(np.array(video_frames, dtype=np.uint8))
            metadata_list.append(
                {"video_name": vid_name, "label": label, "start_frame_index": start_idx}
            )

    # Stack into a single array: Shape (400, n, 224, 224, 3)
    frames_array = np.array(frames_array, dtype=np.uint8)

    # Save to disk
    dataset = {"frames": frames_array, "metadata": metadata_list}
    np.save(output_npy, dataset)
    print(
        f"Successfully processed and saved {len(metadata_list)} videos to {output_npy}."
    )
    print(f"Shape of frames array: {frames_array.shape}")


if __name__ == "__main__":
    # Example usage:
    process_video_list(
        video_list="/media/wizav/Data/data/timesformer_smell_test/data/sampled_video_list.txt",
        output_npy="/media/wizav/Data/data/timesformer_smell_test/data/processed/frames_16/res_448/video_dataset.npy",
        n=16,
        img_size=448,
    )
