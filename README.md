<div align="center">

# 🧠 MRI Brain Tumor Diagnosis System

![Brain Tumor Detection Banner](https://img.shields.io/badge/AI-Medical%20Imaging-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An AI-powered deep learning system for automated brain tumor detection and classification from MRI scans**

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Render-46E3B7?style=for-the-badge)](https://mri-brain-tumor-diagnosis-system.onrender.com/)

[Features](#-key-features) • [Tech Stack](#-tech-stack--libraries) • [Installation](#️-installation) • [Usage](#-usage) • [Deployment](#️-deployment)

---

</div>

## 📋 Overview

This project harnesses the power of **deep learning** to tackle a critical medical challenge: accurately classifying brain tumors from MRI scans. Built using **transfer learning with the renowned VGG16 architecture**, it classifies tumors into four key categories: **glioma**, **meningioma**, **pituitary**, or **no tumor**. 

Leveraging Python and TensorFlow, this system aims to deliver reliable, interpretable AI diagnostics that can support medical imaging workflows.

## 🚀 Key Features

🧬 **Transfer Learning with VGG16**
Leveraging the pre-trained VGG16 convolutional base, freezing most layers to retain powerful learned features, while fine-tuning the top layers to specialize the model for brain tumor MRI classification.

🖼️ **Custom Image Augmentation**
To maximize model robustness despite a limited dataset, brightness and contrast adjustments are applied on-the-fly during training, improving the model's ability to generalize.

🔢 **Dynamic Data Loading & Label Encoding**
Images are efficiently loaded from organized folder structures, shuffled, and labels encoded dynamically for seamless batch training.

⚙️ **Tailored Model Architecture**
The core model stacks the VGG16 base (excluding its classification head) with flattening, dropout layers, and fully connected dense layers topped with softmax for multi-class tumor prediction.

📈 **Thorough Training and Evaluation**
Training uses the Adam optimizer and sparse categorical cross-entropy loss. The system monitors accuracy and loss trends and validates model performance via detailed classification reports, confusion matrices, and ROC/AUC analyses.

💾 **Model Persistence & Practical Inference**
After training, the model is saved (`model.h5`) and can predict tumor type and confidence on new MRI images through an easy-to-use inference function.

## 🧰 Tech Stack & Libraries

<div align="center">

| Technology | Purpose |
|------------|---------|
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) | Deep learning model construction, transfer learning, and training |
| ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) | High-level neural networks API |
| ![Pillow](https://img.shields.io/badge/Pillow-663399?style=flat) | Image preprocessing and augmentation |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical data manipulation |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) | Visualization of training metrics |
| ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat) | Statistical data visualization |
| ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Performance metrics and analytical tools |
| ![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white) | Web application backend |
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Scripting and automation backbone |

</div>

## 🏗️ Model Architecture

```
Input MRI Image (224x224x3)
         ↓
┌────────────────────────┐
│   VGG16 Base Model     │
│   (Pre-trained on      │
│    ImageNet)           │
│   - Conv Layers        │
│   - MaxPooling         │
│   - (Most layers frozen)│
└────────────────────────┘
         ↓
┌────────────────────────┐
│   Flatten Layer        │
└────────────────────────┘
         ↓
┌────────────────────────┐
│   Dropout (0.5)        │
└────────────────────────┘
         ↓
┌────────────────────────┐
│   Dense (256 units)    │
│   ReLU Activation      │
└────────────────────────┘
         ↓
┌────────────────────────┐
│   Dropout (0.5)        │
└────────────────────────┘
         ↓
┌────────────────────────┐
│   Dense (4 units)      │
│   Softmax Activation   │
└────────────────────────┘
         ↓
    Output Classes:
    - Glioma
    - Meningioma
    - Pituitary
    - No Tumor
```

## 🛠️ Development Workflow

1️⃣ **Data Preparation**
- Loaded MRI images from well-structured training and testing folders
- Shuffled datasets and encoded labels dynamically
- Applied brightness and contrast augmentations in real-time during training to boost robustness

2️⃣ **Model Setup**
- Initialized VGG16 pretrained on ImageNet without its top layers
- Froze most layers to preserve pretrained features; unfroze several top layers for targeted fine-tuning
- Added custom classification layers with dropout and dense connections

3️⃣ **Training**
- Used a custom data generator for efficient batch processing
- Trained the model over multiple epochs with the Adam optimizer
- Tracked and visualized accuracy and loss metrics throughout the training cycle

4️⃣ **Evaluation**
- Assessed model performance with classification reports and confusion matrices on test data
- Visualized results through heatmaps and ROC/AUC curves for detailed understanding

5️⃣ **Inference**
- Preprocessed new MRI images and ran predictions with confidence scoring
- Presented results alongside input images for interpretability

## 🖥️ Installation

Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook
- (Optional) Virtual environment

Setup

1. **Clone the repository**
```bash
git clone https://github.com/PasinduSuraweera/MRI-Brain-Tumor-Diagnosis-System.git
cd MRI-Brain-Tumor-Diagnosis-System
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

## 📦 Required Dependencies

Create a `requirements.txt` file with the following packages:

```txt
tensorflow>=2.10.0
keras>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
pillow>=8.3.0
scikit-learn>=0.24.0
opencv-python>=4.5.0
flask>=2.0.0
gunicorn>=20.1.0
```

## 🚀 Usage

Training the Model

1. Open the training notebook in Jupyter
2. Organize your dataset in the following structure:
```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```
3. Run the cells sequentially to train the model
4. Model checkpoints will be saved as `model.h5`

Making Predictions

**In Jupyter Notebook:**
```python
# Load the trained model
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('model.h5')

# Preprocess and predict
def predict_tumor(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    
    return predicted_class, confidence

# Use the function
predict_tumor('path/to/mri_scan.jpg')
```

## ☁️ Deployment

The system is deployed on **Render** to provide seamless web access. The backend is built using **Flask**, serving the model and handling image uploads smoothly. 

🌐 **Live Demo**
Explore the live application here:

<div align="center">

**[https://mri-brain-tumor-diagnosis-system.onrender.com/](https://mri-brain-tumor-diagnosis-system.onrender.com/)**

</div>

Deploying Your Own Instance

1. **Create a Flask Application** (`app.py`)
2. **Configure `requirements.txt`** with all dependencies
3. **Create `Procfile`** for Render:
```
web: gunicorn app:app
```
4. **Deploy to Render**:
   - Connect your GitHub repository
   - Configure build settings
   - Deploy automatically on push

## 📁 Project Structure

```
MRI-Brain-Tumor-Diagnosis-System/
│
├── notebooks/                   # Jupyter notebooks
│   ├── training.ipynb          # Model training pipeline
│   ├── evaluation.ipynb        # Model evaluation & metrics
│   └── inference.ipynb         # Prediction demonstrations
│
├── data/                       # Dataset directory
│   ├── Training/               # Training images
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── pituitary/
│   │   └── notumor/
│   └── Testing/                # Testing images
│       ├── glioma/
│       ├── meningioma/
│       ├── pituitary/
│       └── notumor/
│
├── models/                     # Saved models
│   └── model.h5               # Trained VGG16 model
│
├── app.py                      # Flask web application
├── requirements.txt            # Project dependencies
├── Procfile                    # Render deployment config
└── README.md                   # Project documentation
```

## 🔬 Dataset

This project works with MRI brain scan datasets organized into four categories:
- **Glioma** - A type of tumor that occurs in the brain and spinal cord
- **Meningioma** - A tumor that arises from the meninges
- **Pituitary** - A tumor in the pituitary gland
- **No Tumor** - Healthy brain scans

Recommended Datasets:
- [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [Brain MRI Images for Tumor Detection (Kaggle)](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

**Important**: Ensure you have proper authorization and follow ethical guidelines when using medical data.

## ⚠️ Important Disclaimer

> **This model is not 100% accurate and is trained on a relatively small dataset**, which may impact its performance on new, unseen cases. 
>
> **This system should be viewed as a supportive diagnostic aid rather than a definitive medical tool.** It is designed for research and educational purposes and should **NOT** be used as a substitute for professional medical diagnosis. 
>
> Always consult qualified healthcare professionals for medical decisions.

## 🎯 Conclusion

This MRI Brain Tumor Diagnosis System showcases how **transfer learning with VGG16** can be effectively applied to challenging medical imaging tasks, even when working with limited data. By combining strategic data augmentation, fine-tuning, and comprehensive evaluation, the system achieves promising tumor classification accuracy.

This project lays a solid foundation for future AI-driven diagnostic applications and underscores deep learning's potential to empower healthcare professionals with faster, more informed decision-making.

🔮 Future Directions
- 📈 Expanding the dataset with more diverse MRI scans
- 🏥 Including additional tumor classes and subtypes
- 🔬 Implementing explainable AI techniques (Grad-CAM, attention maps)
- 🏗️ Progressing toward clinical deployment and validation
- ⚡ Exploring more advanced architectures (ResNet, EfficientNet, Vision Transformers)
- 🌍 Multi-center validation studies

## 🤝 Contributing

Contributions are welcome! Whether it's bug fixes, feature additions, or documentation improvements, your help is appreciated.

How to Contribute:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Pasindu Suraweera**

- 🐙 GitHub: [@PasinduSuraweera](https://github.com/PasinduSuraweera)
- 🌐 Live Demo: [MRI Brain Tumor Diagnosis System](https://mri-brain-tumor-diagnosis-system.onrender.com/)

## 🙏 Acknowledgments

- 🏥 Medical imaging research community
- 🤖 TensorFlow and Keras development teams
- 🎓 ImageNet and VGG16 researchers
- 💡 Open-source contributors
- 👨‍⚕️ Healthcare professionals providing guidance and feedback

## 📧 Contact

For questions, feedback, or collaboration opportunities, please open an issue in the repository or reach out through GitHub.

---

<div align="center">

⭐ **If you find this project helpful, please consider giving it a star!** ⭐

**Made for advancing healthcare AI**

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=PasinduSuraweera.MRI-Brain-Tumor-Diagnosis-System)

*Building the future of medical diagnostics, one model at a time* 🚀

</div>
