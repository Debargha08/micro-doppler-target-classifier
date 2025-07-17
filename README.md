# Doppler-Based Dynamic Target Classifier

A deep learning project to classify dynamic motion types (human vs. vehicle) using synthetic Doppler spectrograms.

## 🚀 Overview

This project generates and classifies spectrograms of simulated Doppler signals. It distinguishes between:
- **Human motion** – periodic pattern
- **Vehicle motion** – continuous pattern

A Convolutional Neural Network (CNN) is trained on grayscale spectrograms to achieve over **97% accuracy** on test data.

## 📁 Project Structure

doppler_classification/
├── data/
│ ├── train/
│ │ ├── human/
│ │ └── vehicle/
│ └── test/
│ ├── human/
│ └── vehicle/
├── doppler_classifier.keras
