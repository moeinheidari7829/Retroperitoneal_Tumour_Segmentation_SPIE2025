# Retroperitoneal Tumour Segmentation (ViLU-Net)

ViLU-Net is our SPIE 2025 architecture built on nnU-Net v2, embedding Vision-LSTM (ViL) blocks in both encoder and decoder. Two variants are provided: 2D and 3D full-resolution.

## Data Availability
Due to privacy reasons, the retroperitoneal dataset used in our experiments cannot be shared. The code is fully usable on any dataset organized in nnU-Net format. Example public datasets and instructions are listed in `data/README.md`.

## Architecture at a Glance
![ViLU-Net Figure](data/figure.png)

## Quick Start
1) Create environment (example): `conda create -n vilunet python=3.10 -y && conda activate vilunet`
2) Install PyTorch matching your CUDA: `pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118`
3) Install this repo in editable mode:
   ```bash
   cd ViLU-Net
   pip install -e .
   ```
4) Preprocess data with nnU-Net v2:
   ```bash
   nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
   ```

## Training
Use the unified trainer `nnUNetTrainerViLUNet` for both 2D and 3D.
```bash
# 2D
nnUNetv2_train DATASET_ID 2d all -tr nnUNetTrainerViLUNet -lr 5e-3 -bs 2

# 3D full-res
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerViLUNet -lr 5e-3 -bs 1
```
Adjust `-bs` and `-lr` to your hardware and dataset.

## Inference
```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c {2d|3d_fullres} -f all -tr nnUNetTrainerViLUNet --disable_tta
```

## Notes

- Credits: We appreciate the xLSTM-UNet authors for releasing their Vision-LSTM components, which this work builds upon.
