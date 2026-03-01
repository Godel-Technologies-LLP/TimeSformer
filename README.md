Overview
This pipeline extracts video embeddings using TimeSformer across multiple controlled variants:
- Temporal attention (Alpha)
- Resolution scaling (Beta)
- Temporal consistency (Gamma)

Outputs are stored as '.npy' embeddings for downstream analysis.

1. Environment Tested:

OS: Ubuntu 20.04
Python: 3.7
CUDA: 11.x
GPU: NVIDIA V100 (32GB recommended for large variants)
PyTorch: Compatible with CUDA 11

2. Build Instructions:

Step 1: Create Environment
conda create -n timesformer python=3.7 -y
conda activate timesformer

Step 2: Install Dependencies
pip install torchvision
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install simplejson einops timm psutil scikit-learn opencv-python tensorboard
conda install av -c conda-forge

Step 3: Clone and Build
git clone https://github.com/facebookresearch/TimeSformer
cd TimeSformer
python setup.py build develop

3. Execution Examples:

Extract Embeddings (Default Configuration)
python /home/wizav/src/TimeSformer/tools/deepfake_detector_train.py   --cfg configs/Kinetics/TimeSformer_divST_8x32_224_TEST.yaml   DATA.PATH_TO_DATA_DIR /media/wizav/Data/data/timesformer_smell_test/data   MODEL.NUM_CLASSES 2   TIMESFORMER.PRETRAINED_MODEL /media/wizav/Data/data/timesformer_smell_test/models/backbones/TimeSformer_divST_8x32_224_K400.pyth   NUM_GPUS 1   TEST.CHECKPOINT_FILE_PATH ""   DATA.PATH_LABEL_SEPARATOR ","   TEST.BATCH_SIZE 8

Variants Overview:

Alpha: Temporal Attention
space_only → ignores temporal information
divided_space_time → default (temporal + spatial split)
joint_space_time → full joint attention
Beta: Resolution
224 → standard baseline
448 → high-resolution (better spatial detail)

Gamma: Frame Length
8 → short temporal window
32 → medium temporal context
96 → long-range temporal modeling
Attention Heatmap Extraction
outputs = model(video_pixels, output_attentions=True)
outputs.attentions returns attention maps for visualization

4. Troubleshooting:

CUDA Out of Memory
Reduce batch size:
TEST.BATCH_SIZE = 1
Use lower resolution (224 instead of 448)

Model Not Loading
Verify pretrained model path:
TIMESFORMER.PRETRAINED_MODEL=/absolute/path/to/model.pyth

Python Version Issues
Ensure Python 3.7 is used

Slow Inference
Ensure GPU is being used:
nvidia-smi

5. Validation:

Quick Sanity Check:

import torch
from timesformer.models.vit import TimeSformer

model = TimeSformer(
    img_size=224,
    num_classes=400,
    num_frames=8,
    attention_type='divided_space_time'
)

dummy = torch.randn(2, 3, 8, 224, 224)
out = model(dummy)

print(out.shape)
Expected Output
torch.Size([2, 400])
Embedding Output Check

After running extraction, verify:

embeddings/
├── alpha_temporal/
├── beta_resolution/
├── gamma_consistency/

Notes:
Use 224 resolution for debugging and speed
Use 448 resolution for fine spatial artifacts
Use 96 frames for long-term temporal inconsistencies
Default (divided_space_time) provides best balance