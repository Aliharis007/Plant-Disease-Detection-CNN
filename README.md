# 🌿 Plant Disease Detection using CNN 🧠🦠

Welcome to the **Plant Disease Detection** project — a deep learning-based solution that uses Convolutional Neural Networks (CNNs) to classify plant leaf images as **healthy** or **diseased**. This tool is aimed at helping farmers 🌾 and researchers identify plant diseases early and take preventive action.

---

## Abstract
This project focuses on the development of an AI-based plant disease detection system using Convolutional Neural Networks (CNNs). Specifically, it aims to classify potato leaf images into three categories: healthy, early blight, and late blight. Utilizing the PlantVillage dataset and built with TensorFlow and Keras, the model underwent rigorous preprocessing, including resizing, normalization, and data augmentation to enhance generalization. A custom CNN architecture was trained on over 2,000 images, achieving a test accuracy of 99.91% and demonstrating exceptional classification performance. The results, supported by confusion matrix and classification reports, show the model’s potential for real-world agricultural deployment. Future directions include incorporating transfer learning and real-time mobile or web-based implementation for accessible plant health diagnostics.
## 📸 Project Overview

💡 **Goal:** Build a deep learning model that can accurately detect and classify plant diseases from leaf images.  
🧠 **Approach:** Train a CNN on thousands of labeled images to recognize patterns associated with various plant diseases.  
📊 **Tools & Libraries:**  
- Python 🐍  
- TensorFlow / Keras 🔧  
- OpenCV 📷  
- Matplotlib 📈  
- NumPy ➕  
- Scikit-learn 📚  

---

## 🗂️ Dataset Info

We used the **PlantVillage Dataset** — a popular dataset available on Kaggle including healthy and various diseases.

📌 **Download Dataset:**  
👉 [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

📦 **Contents:**
- High-quality leaf images of Potato 🥔 crops
- Each image is labeled with the crop and disease type (or marked as healthy).
- Structured into folders by class (e.g., `Potato___Late_blight`, `Potato___Early_blight`, etc.)

---

## 🔧 Setup Instructions

### 1. 📥 Clone the Repository

```bash
git clone https://github.com/Aliharis007/Plant-Disease-Detection-CNN.git
cd Plant-Disease-Detection-CNN
````

### 2. 🛠️ Create Virtual Environment (Recommended)

```bash
# Create and activate venv
python -m venv venv
venv\Scripts\activate   # For Windows
# OR
source venv/bin/activate   # For macOS/Linux
```

### 3. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. 🧾 Download Dataset

* Download the dataset from the Kaggle link above.
* Extract it and place the dataset folder into the `/data` directory inside this repo.

### 5. 🚀 Train the Model

```bash
python train_model.py
```

The training script will:

* Preprocess the data
* Train a CNN model
* Save the trained model in `/model/` folder

---

## 🧪 Run Inference / Prediction

To predict disease from a custom leaf image:

```bash
python predict_disease.py --image path_to_your_image.jpg
```

The script will:

* Load the trained model
* Preprocess the input image
* Predict and display the disease class 🎯

---

## 📁 Project Structure

```bash
Plant-Disease-Detection-CNN/
│
├── data/                     # Dataset directory (PlantVillage images)
├── model/                    # Saved trained model
├── train_model.py            # CNN model training script
├── predict_disease.py        # Prediction / Inference script
├── requirements.txt          # All dependencies
└── README.md                 # This file 📘
```

---

## 💡 Future Work

* Add GUI/Web Interface for real-time use 🌐
* Integrate with mobile camera input 📱
* Improve accuracy using transfer learning (e.g., MobileNet, ResNet) 📈
* Deploy as a Flask or Streamlit app 🚀

---

## 🤝 Contributing

Feel free to fork this repo, improve the code, and submit pull requests. All contributions are welcome! 🌍

---

## 🙏 Acknowledgements

* 🧠 [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) — Kaggle
* ❤️ TensorFlow / Keras community
* 📷 OpenCV, NumPy & Matplotlib teams


## 🔗 Connect

Made with 💚 by [Ali Haris](https://github.com/Aliharis007)


