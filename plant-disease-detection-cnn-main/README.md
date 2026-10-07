# Plant Disease Detection using CNN

## What is this project?
This project builds a system that looks at a photo of a plant leaf and identifies whether it has a disease — and which one. Farmers often struggle to tell diseases apart just by looking, so this automates that process using deep learning.

## How It Works

**1. Dataset**
Used the PlantVillage dataset — about 20,000 leaf photos of pepper, potato, and tomato plants, each labeled with its disease (or marked healthy).

**2. Preprocessing**
Before feeding images to the model, all images were resized to the same size (224×224) and pixel brightness values were scaled down to a 0–1 range (normalization). This helps the model train faster and more reliably.

**3. Data Augmentation**
Applied random flips, rotations, and zooms to training images. This stops the model from memorizing exact photos and forces it to learn actual disease patterns instead — making it generalize better to new images.

**4. Model — Transfer Learning**
Instead of building a CNN completely from scratch, this project uses transfer learning: MobileNetV2, a CNN already trained on 1.4 million general images, is reused for its learned knowledge of shapes, edges, and textures. That part is frozen, and a small new classification layer is added on top to learn the 15 specific disease categories in this dataset.

**5. Training**
Trained for 10 epochs — meaning the model looked at all training images 10 times, adjusting itself each time to make fewer mistakes.

## Results
- **Validation Accuracy: 89.3%** — on leaf images the model had never seen before, it correctly identified the disease about 9 times out of 10.
- Checked a confusion matrix to see which diseases it sometimes confused — mainly a few tomato diseases that look visually similar.
- Tested on a real photo (not from the dataset) — correctly predicted the disease with **93.81% confidence**.

## Dataset
- **Source:** [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Size:** 20,638 images, 15 classes
- **Split:** 80% training (16,511 images) / 20% validation (4,127 images)

## Tech Stack
- Python, TensorFlow / Keras
- Google Colab (GPU-accelerated training)
- scikit-learn, Matplotlib, Seaborn (evaluation & visualization)

## Pipeline
```
Image → Resize/Normalize → Augment (training only) → MobileNetV2 (frozen) 
→ Custom Classifier → Predicted Disease + Confidence
```
## Limitations
The model was trained only on Pepper, Potato, and Tomato leaf images (15 classes). 
When given leaves from other plants (e.g., banana), it still forces a prediction 
into one of its known 15 categories rather than recognizing the input as 
out-of-scope — since the model has no "unknown" class. This is a common 
limitation of standard image classifiers and would require either a broader 
training dataset or an added rejection/confidence-threshold mechanism to fix.
