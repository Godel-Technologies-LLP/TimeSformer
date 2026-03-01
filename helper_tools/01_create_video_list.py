import random
from pathlib import Path

def create_video_list(real_dir, fake_dir, output_txt="video_list.txt", samples=200):
    # Get all mp4 files
    real_vids = list(Path(real_dir).glob('*.mp4'))
    fake_vids = list(Path(fake_dir).glob('*.mp4'))

    # Sample randomly
    if len(real_vids) >= samples:
        real_vids = random.sample(real_vids, samples)
    if len(fake_vids) >= samples:
        fake_vids = random.sample(fake_vids, samples)

    # Resolve absolute paths and assign labels (0=Real, 1=Fake)
    data = [(str(vid.resolve()), 0) for vid in real_vids] + \
           [(str(vid.resolve()), 1) for vid in fake_vids]
    
    # Shuffle so real and fake are mixed
    random.shuffle(data)

    # Write to a space-separated text file
    with open(output_txt, 'w') as f:
        for filepath, label in data:
            f.write(f"{filepath} {label}\n")
            
    print(f"Saved {len(data)} video paths to {output_txt}")

if __name__ == "__main__":
    # Example usage:
    create_video_list("/media/wizav/Data/data/celeb/Celeb-real", "/media/wizav/Data/data/celeb/Celeb-synthesis", output_txt="/media/wizav/Data/data/timesformer_smell_test/data/sampled_video_list.txt")    
