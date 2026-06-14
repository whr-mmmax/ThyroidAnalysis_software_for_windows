# Thyroid Image Analysis Software
This project is a **thyroid image analysis tool** developed for the *Digital Image Processing* course homework of Hunan University (HNU). HNUer can use this project for your own course assignments.

## Download Link
Baidu Netdisk:
- Link: https://pan.baidu.com/s/1Xe8q_yZkFP0_exsp7UK3KQ
- Extract Code: `5whr`
- File: `dist.zip`

## Usage Guide
### 1. Run with Pre-trained Model (No Retraining Required)
1. Download and unzip `dist.zip`.
2. Prepare model files: `best_classifier.pth` and `best_unet.pth`.
3. Create a folder named `checkpoints`, then put the two `.pth` files inside it.
4. Ensure the `checkpoints` folder and `ThyroidAnalysis.exe` are in the **same directory**.
5. Run `ThyroidAnalysis.exe` directly.

### 2. Retrain Model with Custom Dataset
1. Open the `config` file.
2. Change the `DATA_ROOT` value to your local dataset path.
3. Follow the project code to retrain the model.

## Notice for HNUer
This project is for the **Digital Image Processing** course homework. You can use it for your own course projects.
