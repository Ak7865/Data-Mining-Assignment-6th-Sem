# Assignment 4 — Skin Cancer Detection (CNN - MobileNetV2)

## 🎯 Objective
Build a **Deep Learning model** to classify skin images into:

- **Melanoma (Cancer)**
- **Not Melanoma (Non-Cancer)**

using **Transfer Learning (MobileNetV2)**.

---

## 📂 Dataset

Dataset folder structure:
DermMel/
├── train_sep/
│ ├── Melanoma/
│ └── NotMelanoma/
├── valid/
└── test/


- Images are loaded using `flow_from_directory()`
- Classes are automatically detected from folder names

---

## ⚙️ Implementation Logic

### 🔹 Load Dataset
- Load image dataset using `ImageDataGenerator`
- Apply preprocessing using:
preprocess_input (MobileNetV2)


---

### 🔹 Data Augmentation
Applied on training data:

- Rotation
- Zoom
- Horizontal Flip

---

### 🔹 Class Weights
To handle class imbalance:
compute_class_weight()

---

### 🔹 Model Architecture

- Base Model: **MobileNetV2 (Pretrained on ImageNet)**
- Custom Layers:
  - GlobalAveragePooling
  - Dense (128, ReLU)
  - BatchNormalization
  - Dropout (0.4)
  - Output Layer (Softmax - 2 classes)

---

### 🔹 Training Strategy

#### Stage 1 — Transfer Learning
- Freeze base model
- Train only top layers

#### Stage 2 — Fine Tuning
- Unfreeze last 30 layers
- Reduce learning rate
- Improve model performance

---

### 🔹 Callbacks

- EarlyStopping → Stops training when no improvement  
- ReduceLROnPlateau → Reduces learning rate  
- ModelCheckpoint → Saves best model  

---

### 🔹 Model Saving

- `cancer_model.h5` → Final trained model  
- `best_model.h5` → Best model during training  

---

### 🔹 Evaluation

Metrics used:

- Confusion Matrix  
- Classification Report  

---

### 🔹 Accuracy Formula
Accuracy = (TRUE POSITIVE + TRUE NEGATIVE) / TOTAL PREDICTIONS


---

## 🚀 Prediction (Predict.py)

### 🔹 Input
- Image file path:
DermMel/test/Melanoma/AUG_0_11.jpeg


---

### 🔹 Prediction Logic

- Load trained model (`cancer_model.h5`)
- Resize image to 224×224  
- Apply preprocessing  
- Predict class  

---

### 🔹 Output
Prediction: Melanoma
Confidence: 97.45%


---

## 📷 Sample Output

![Output 1](https://github.com/Ak7865/Data-Mining-Assignment-6th-Sem/blob/main/Assignment-4/image.png)

![Output 2](https://github.com/Ak7865/Data-Mining-Assignment-6th-Sem/blob/main/Assignment-4/image1.png)

---

---

## 📷 Confusion Matrics

![Output](https://github.com/Ak7865/Data-Mining-Assignment-6th-Sem/blob/main/Assignment-4/confusion_matrix.png)


---

## ✅ Features

- Deep Learning model using CNN  
- Transfer Learning (MobileNetV2)  
- Data Augmentation  
- Class imbalance handling  
- Model saving and loading  
- Image-based prediction  

---

## ⚠️ Limitations

- Requires good dataset quality  
- Needs GPU for faster training  
- Overfitting possible on small datasets  

---

## 🔮 Future Improvements

- Add Grad-CAM visualization  
- Use larger datasets  
- Build web interface  
- Try advanced models (EfficientNet)  

---

## 👨‍💻 Author

**Syed Akhter Hussain**  
B.Tech CSE Student  
Barak Valley Engineering College  

---