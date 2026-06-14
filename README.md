# Aerial GCP Pose Estimation - Skylark Drones Assignment

## Overview

This project addresses the problem of automatically detecting Ground Control Point (GCP) markers in aerial drone imagery.

The objective is to:

1. Predict the pixel coordinates `(x, y)` of the center of the GCP marker.
2. Classify the marker shape into one of the following classes:
   - Cross
   - Square
   - L-Shape

The solution is implemented using PyTorch and EfficientNet-B0 in a multi-task learning setup.

---

## Dataset Exploration

Before model development, Exploratory Data Analysis (EDA) was performed on the provided training dataset.

### Dataset Statistics

- Total samples in annotation file: 1000
- Valid labeled samples used for training: 996
- Samples missing shape labels: 4

### Shape Distribution

| Shape   | Count |
| ------- | ----- |
| L-Shape | 491   |
| Square  | 328   |
| Cross   | 177   |

The dataset is moderately imbalanced, with Cross markers being the least represented class.

### Image Resolutions

Two image resolutions were observed:

- 4096 × 2730
- 4096 × 3068

### Marker Characteristics

Observations from manual inspection:

- Markers are clearly visible in most images.
- Marker size is approximately 40–50 pixels.
- Marker locations are distributed throughout the image.
- Marker shapes are visually distinguishable.
- No obvious annotation errors were found during inspection.

---

## Model Architecture

A multi-task learning architecture was implemented using EfficientNet-B0 as a shared feature extractor.

### Backbone

- EfficientNet-B0 (pretrained ImageNet weights)

### Output Heads

#### Coordinate Regression Head

Predicts:

- x_norm
- y_norm

Coordinates are normalized to the range [0, 1].

#### Shape Classification Head

Predicts one of:

- Cross
- Square
- L-Shape

---

## Training Strategy

### Image Preprocessing

Original images are resized to:

768 × 768

Coordinates are normalized using the original image dimensions before resizing.

### Data Augmentation

Albumentations was used with:

- Horizontal Flip
- Vertical Flip
- RandomRotate90
- Random Brightness/Contrast

### Loss Functions

Coordinate Regression:

- SmoothL1Loss

Shape Classification:

- Weighted CrossEntropyLoss

Class weights were used to reduce the effect of class imbalance.

### Optimizer

- AdamW
- Learning Rate: 1e-4

### Training Configuration

- Batch Size: 8
- Epochs: 30
- Train/Validation Split: 80/20

---

## Validation Results

Best Validation Loss:

0.0349

The model achieved its best validation performance around epoch 10 and later showed mild signs of overfitting.

The best checkpoint was automatically saved as:

best_model.pth

---

## Inference

To generate predictions:

```bash
python src/inference.py
```

Output:

```text
predictions.json
```

The generated file follows the exact structure required by the assignment specification.

---

## Project Structure

```text
.
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
│
├── README.md
├── requirements.txt
└── predictions.json
```

---

## Challenges Encountered

### 1. Incorrect Assumption About Image Resolution

Initially, image dimensions were assumed to match the assignment description.

During EDA, it was discovered that the actual dataset contained images of size:

- 4096 × 2730
- 4096 × 3068

The coordinate normalization pipeline was updated to use the original dimensions of each image instead of hardcoded values.

### 2. Missing Labels

Four samples in the annotation file did not contain the `verified_shape` field.

These samples were excluded from training to avoid introducing noisy targets.

### 3. Hardware Constraints

Initial experiments were attempted locally, but training high-resolution aerial imagery on the available hardware was impractical.

To ensure reproducible training, the final model was trained on Google Colab using a Tesla T4 GPU.

### 4. Model Design Decisions

An early idea was to use a heatmap-based keypoint localization approach.

After analyzing the dataset characteristics and considering implementation complexity, a simpler and more deployable multi-task EfficientNet-B0 architecture was chosen.

This approach provided a good balance between localization accuracy, training efficiency, and engineering simplicity.

---

## Assumptions

- Samples without a valid `verified_shape` label were ignored.
- Normalized coordinates are sufficient for training and are converted back to pixel coordinates during inference.
- The provided annotations are assumed to be correct unless obvious inconsistencies are observed.

---

## Author

- Niteesh Ram
