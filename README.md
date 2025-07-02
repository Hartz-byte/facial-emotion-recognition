![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-API-red?logo=keras)
![EfficientNet](https://img.shields.io/badge/EfficientNet-B0-green)
![Model](https://img.shields.io/badge/Model-EfficientNetB0-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-FER2013-blue)


# Facial Emotion Recognition with EfficientNetB0
This project implements a deep learning-based Facial Emotion Recognition (FER) system using EfficientNetB0 as the backbone model, trained and fine-tuned on the popular FER2013 dataset.

---

## Project Overview
- Dataset: FER2013 — A labeled dataset of 48x48 grayscale facial images categorized into 7 emotion classes.
- Model: Transfer learning using EfficientNetB0 with fine-tuning.
- Input Format: Grayscale images (converted to 3 channels).
- Target Labels: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

---

## Features
- Facial expression classification using EfficientNetB0
- Extensive data augmentation
- Class imbalance handling with class_weight
- Label smoothing regularization
- Transfer learning + fine-tuning
- Visualizations: training accuracy, confusion matrix, and class distribution

---
## Dataset Info
FER2013 has 35,887 grayscale images (48x48) labeled as:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise
You can download the dataset from here

---

## Results
- Accuracy:
  | Phase                                            | Accuracy   |
  | ------------------------------------------------ | ---------- |
  | Initial Training (30 epochs)                     | \~70.8%    |
  | After Fine-Tuning (EfficientNet last 100 layers) | \~71%      |
  | Final Test Accuracy                              | **67.19%** |

- Classification Report:
  | Emotion              | Precision | Recall | F1-score |
  | -------------------- | --------- | ------ | -------- |
  | Angry                | 0.56      | 0.63   | 0.59     |
  | Disgust              | 0.65      | 0.66   | 0.65     |
  | Fear                 | 0.54      | 0.44   | 0.48     |
  | Happy                | 0.89      | 0.86   | 0.88     |
  | Neutral              | 0.62      | 0.68   | 0.65     |
  | Sad                  | 0.56      | 0.50   | 0.53     |
  | Surprise             | 0.74      | 0.84   | 0.79     |
  | **Overall Accuracy** |           |        | **67%**  |

---

## Model Architecture
Input: (96, 96, 1) grayscale
→ Conv2D (3x3) → (96, 96, 3)
→ EfficientNetB0 (pretrained on ImageNet)
→ GlobalAveragePooling2D
→ Dropout (0.4)
→ Dense (7, softmax)

- Loss: CategoricalCrossentropy(label_smoothing=0.1)
- Optimizer: Adam, fine-tuned with LR = 3e-6

---

## ⭐️ Give it a Star

If you found this repo helpful or interesting, please consider giving it a ⭐️. It motivates me to keep learning and sharing!

---

