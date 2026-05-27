# Retinal Image Analysis for Multi-Disease Detection – Sub-challenge 1

**Binary screening of retinal fundus images (Normal vs Abnormal) using EfficientNet-B0 and PyTorch**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Resources

- **Google Slides (Project Presentation):**  
https://docs.google.com/presentation/d/1ofTQg2fMlezXq0-pqgclbLFZaLi-vph1vLihPL9YliU/edit?usp=sharing

- **Kaggle Notebook (Full Training & Evaluation):**  
https://www.kaggle.com/code/bikramchapagain/retinaldiseaseclassificationusingefficientnet

---

##  Overview

This repository contains our solution for **Sub-challenge 1** of the *Retinal Image Analysis for Multi-Disease Detection (RIADD)* challenge, organised in conjunction with **ISBI 2021**.

The goal is to automatically classify retinal fundus images into:

- **Normal (0)** – no retinal pathology  
- **Abnormal (1)** – presence of at least one of 45 retinal diseases  

The evaluation metric is **AUC**.

---

##  Problem Statement

Manual retinal disease screening is time-consuming and requires specialist expertise. Early detection of retinal abnormalities is crucial to prevent vision impairment and blindness.

This project develops a robust deep-learning pipeline that can screen retinal fundus images automatically, providing a probability of abnormality for each image.

---

## Dataset

We use the **Retinal Fundus Multi-disease Image Dataset (RFMiD)**.

| Set        | Number of Images |
|------------|------------------|
| Training   | 1920             |
| Validation | 640              |
| Test       | 640              |
| **Total**  | **3200**         |

### Dataset Details

- **Images:** Colour fundus photographs (PNG) from three different cameras  
- **Diseases:** 46 distinct retinal pathologies  
- **Labels:** Binary ground truth (`0` = normal, `1` = abnormal)

---

##  Methodology

### 1. Transfer Learning with EfficientNet-B0

We use an **EfficientNet-B0** backbone pre-trained on ImageNet. The original classifier head is replaced with:

```python
nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(1280, 1)
)
```

---

### 2. Dataset Setup


After downloading, extract the dataset and organise it in the following structure:

```text
data/
└── RFMiD/
    ├── train/
    ├── val/
    ├── test/
   
```

---

### 3. Training

#### Training Features

- Transfer learning using EfficientNet-B0  
- Automatic validation after each epoch  
- Early stopping to prevent overfitting  
- Learning rate scheduling  
- Best model checkpoint saving  

The trained model will be saved in:

```text
models/best_model.pth
```

Run training using:

```bash
python src/train.py --data_dir data/RFMiD --batch_size 32 --epochs 15
```

---

### 4. Generate Submission

After training and inference, the submission file will automatically be generated inside:

```text
submission/teamName_results.csv
```

#### Submission Format

```text
ImageID    Probability    Disease_Risk
```

Example:

```text
1001.png    0.9821    1
1002.png    0.0412    0
```

Where:

- `Probability` = predicted abnormality score  
- `0` = Normal image  
- `1` = Abnormal image  

---

## Results

| Metric | Value |
|--------|-------|
| Validation AUC | 0.9405 |
| Training AUC | 0.9671 |
| Overfitting Gap | 0.0266 |
| Optimal Threshold | 0.694 |

### Performance Metrics

- **Accuracy:** 93.9%  
- **Sensitivity:** 95.0%  
- **Specificity:** 91.6%  
- **Precision:** 96.0%  
- **F1-score:** 0.955  

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├──-sourcecode.ipynb
├── src/
├── figures/
├── submission/
│   └── bikky_results.csv
└── models/
    └── best_model.pth
```

---

##  References

1. Pachade, S., et al. (2021). RFMiD: Retinal Fundus Multi-disease Image Dataset.  
2. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for CNNs.

---

##  Contributors

- Bikram Chapagain

---


