# Retinal Image Analysis for Multi-Disease Detection (RIADD)
## Sub-Challenge 1: Disease Screening using PyTorch

## Overview
This project focuses on automated retinal disease screening using deep learning on the RFMiD (Retinal Fundus Multi-Disease Image Dataset) dataset. The goal of Sub-Challenge 1 is to classify retinal fundus images into:

- Normal (0)
- Abnormal (1)

This is a binary classification problem where the model predicts the probability of a retinal image being abnormal.

The implementation is developed using PyTorch and transfer learning-based Convolutional Neural Networks (CNNs).

---

# Problem Statement

Manual retinal disease screening requires expert ophthalmologists and is time-consuming. Early detection of retinal abnormalities is essential to prevent vision impairment and blindness.

This project aims to develop a robust deep learning model capable of automatically screening retinal fundus images into normal or abnormal categories.

---

# Dataset

Dataset Used:
- RFMiD (Retinal Fundus Multi-Disease Image Dataset)

Total Images:
- 3200 retinal fundus images

Dataset Split:
- Training Set: 1920 images
- Validation Set: 640 images
- Test Set: 640 images

Image Format:
- PNG

Image Sources:
Images were captured using:
- Kowa VX-10α
- TOPCON 3D OCT-2000
- TOPCON TRC-NW300

---

# Disease Categories

The RFMiD dataset contains 45 retinal diseases/pathologies including:
- Diabetic Retinopathy (DR)
- Age-related Macular Degeneration (ARMD)
- Optic Disc Cupping (ODC)
- Retinitis Pigmentosa (RP)
- Macular Hole (MHL)
- Retinal Vein Occlusion
- Hemorrhagic Retinopathy
- and many others

For Sub-Challenge 1:
- Any disease is labeled as Abnormal (1)
- Healthy retina is labeled as Normal (0)

---

# Ground Truth

Ground truth labels were assigned by two expert retinal specialists.

Labels:
- 0 → Normal
- 1 → Abnormal

---

# finding in subchallage -1 
https://docs.google.com/presentation/d/17qo8qwU-ZhJBF0REOaJ85aeGB8hetMyEHYqzez-UR94/edit?usp=sharing
