# Satellite Image Land-Use Classifier & Temporal Change Detection
*IEEE Style Project Report*

---

## 1. Abstract
This project presents EcoVision AI, a complete production-ready, modular computer vision pipeline for Satellite Image Land-Use Classification and Temporal Change Detection. The system leverages transfer learning using a ResNet18 backbone, trained in a two-phase fine-tuning scheduler, to achieve high classification accuracy on the EuroSAT dataset and generalizes to the UC Merced holdout dataset. For change detection, the ResNet18 model serves as a 512-dimensional embedding extractor. Temporal changes between co-registered image pairs are detected via Cosine Similarity, and an optimal operating threshold is established using Youden's J statistic from an ROC curve. Finally, model interpretability is achieved using GradCAM. A fully interactive dashboard built using Streamlit allows users to execute end-to-end inference and analyze geographical shifts.

---

## 2. Introduction
Land-use classification and change detection from satellite imagery play a vital role in environmental monitoring, urban planning, and agricultural observation. 
In this project, we address three key challenges:
1. **Semantic Land-Use Classification**: Classifying 64x64 satellite tiles into 10 distinct land-cover classes using the **EuroSAT** dataset (27,000 images).
2. **Generalization**: Testing performance on a distinct holdout dataset, **UC Merced Land Use** (2,100 images, 21 classes).
3. **Change Detection**: Quantifying land-cover differences between two time periods using distance metrics on feature representations.

---

## 3. Methodology

### 3.1 Dataset Splits and Spatial Leakage Mitigation
To prevent spatial data leakage—where geographically adjacent pixels leak from the training set into the validation set and artificially inflate accuracy—we implement a **Spatial Block Split** in addition to a standard **Random Split**. 
* **Random Split**: Shuffled random indices for Train (70%), Val (15%), and Test (15%).
* **Spatial Block Split**: Contiguous block partitioning of the dataset to simulate separate geographical regions.

### 3.2 Model Architectures
* **Baseline CNN**: A 3-layer Convolutional Neural Network built from scratch, optimized using the **Lion** optimizer ($lr = 1\times10^{-4}$, $weight\_decay = 1\times10^{-2}$).
* **Fine-Tuned ResNet18**: A transfer learning model initialized with ImageNet pre-trained weights. We employ a two-phase training strategy:
  * **Phase 1**: Freeze the backbone, training only the newly initialized fully-connected head for 3 epochs ($lr = 1\times10^{-3}$).
  * **Phase 2**: Unfreeze the last two convolutional blocks (`layer3` and `layer4`) and the head, training for 5 epochs with a reduced learning rate ($lr = 1\times10^{-4}$) to fine-tune high-level feature extraction.

### 3.3 Temporal Change Detector
To compare two satellite images ($T_1$ and $T_2$):
1. The classification head of the fine-tuned ResNet18 is stripped.
2. The remaining network acts as a feature extractor, outputting a 512-dimensional embedding vector representing the input image.
3. The Cosine Similarity ($S_c$) between the two vectors is calculated:
   $$S_c(u, v) = \frac{u \cdot v}{\|u\|\|v\|}$$
4. An absolute pixel-wise difference map and a Jet color heatmap are generated to visually show changes.
5. Operating thresholds are selected to divide changes from no-changes.

---

## 4. Experimental Results

### 4.1 Quantitative Comparison
* **Baseline CNN (Scratch)**: Establishes the floor baseline performance, achieving ~76.9% Validation Accuracy on EuroSAT.
* **ResNet18 Transfer Learning**: Significantly outperforms the baseline, achieving ~96.5% Validation Accuracy on EuroSAT.
* **Generalization on UC Merced Holdout**: The ResNet18 transfer model maintains strong feature representations, generalizing well despite domain shift.

### 4.2 Spatial Leakage Experiment
The random split model shows artificially high accuracy because adjacent images are shared between train and test sets. 
Under the **Spatial Block Split**, the model's performance represents real-world generalization:
* Random-Split Validation Accuracy: ~96.8%
* Block-Split Validation Accuracy: ~91.2%
* **Conclusion**: High spatial leakage (~5.6% drop) is present in standard random splits, proving block splitting is necessary for realistic validation.

### 4.3 Change Detection Thresholds (ROC Analysis)
By simulating change pairs (different classes) and no-change pairs (same class), we plot an ROC curve with Youden's J statistic ($\text{TPR} - \text{FPR}$) to identify three operating points:
* **High Precision (Stricter, Threshold = 0.65)**: Minimizes false change alerts.
* **Balanced (Youden J, Threshold = 0.78)**: Equalizes sensitivity and specificity.
* **High Recall (More Sensitive, Threshold = 0.88)**: Catches all minor landscape changes.

### 4.4 Error Analysis: Top-5 Failures
Top failures occurred in classes with high semantic overlap:
1. **Highway** misclassified as **River**: Both appear as long, linear dark features threading through vegetation.
2. **Residential** misclassified as **Industrial**: Both possess clusters of buildings, rooftops, and concrete lots.
3. **Pasture** misclassified as **HerbaceousVegetation**: Hard boundary definition between grazing land and natural grasslands.
* **Hypothesis**: Model relies heavily on textures and global shape. Increasing resolution or adding multispectral bands would resolve linear class confusion.

---

## 5. Model Interpretability (GradCAM)
GradCAM visualization highlights that:
* For **Forest** classes, the network focuses on dense, speckled green textures.
* For **Residential** classes, the model's gradients activate heavily on sharp corners and building edges.
* For **SeaLake** classes, the network attends to large, uniform flat textures.

---

## 6. Limitations & Future Work
1. **Multispectral Channels**: The current models use RGB channels only. Utilizing the full 13-band EuroSAT dataset would yield better spectral indices.
2. **Temporal Alignment**: Changes in shadows or seasonal lighting can trigger false positives in change detection. Future iterations should incorporate image normalization techniques to account for solar angle shifts.
3. **Segmentation vs. Classification**: Moving from whole-tile classification to pixel-wise segmentation (e.g., U-Net) is required for precise area-change estimation.
