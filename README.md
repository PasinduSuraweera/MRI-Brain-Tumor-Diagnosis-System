# 🧠 MRI Brain Tumor Diagnosis System

This project harnesses the power of deep learning to tackle a critical medical challenge: accurately classifying brain tumors from MRI scans. Built using **transfer learning** with the renowned **VGG16** architecture, it classifies tumors into four key categories: **glioma**, **meningioma**, **pituitary**, or **no tumor**. Leveraging Python and TensorFlow, this system aims to deliver reliable, interpretable AI diagnostics that can support medical imaging workflows.

---

## 🚀 Key Features

- 🧬 **Transfer Learning with VGG16**  
  I leveraged the pre-trained VGG16 convolutional base, freezing most layers to retain powerful learned features, while fine-tuning the top layers to specialize the model for brain tumor MRI classification.

- 🖼️ **Custom Image Augmentation**  
  To maximize model robustness despite a limited dataset, brightness and contrast adjustments are applied on-the-fly during training, improving the model’s ability to generalize.

- 🔢 **Dynamic Data Loading & Label Encoding**  
  Images are efficiently loaded from organized folder structures, shuffled, and labels encoded dynamically for seamless batch training.

- ⚙️ **Tailored Model Architecture**  
  The core model stacks the VGG16 base (excluding its classification head) with flattening, dropout layers, and fully connected dense layers topped with softmax for multi-class tumor prediction.

- 📈 **Thorough Training and Evaluation**  
  Training uses the Adam optimizer and sparse categorical cross-entropy loss. I monitor accuracy and loss trends and validate model performance via detailed classification reports, confusion matrices, and ROC/AUC analyses.

- 💾 **Model Persistence & Practical Inference**  
  After training, the model is saved (`model.h5`) and can predict tumor type and confidence on new MRI images through an easy-to-use inference function.

---

## 🧰 Tech Stack & Libraries

- **TensorFlow & Keras** — Deep learning model construction, transfer learning, and training  
- **PIL (Pillow)** — Image preprocessing and augmentation  
- **NumPy** — Numerical data manipulation  
- **Matplotlib & Seaborn** — Visualization of training metrics and evaluation insights  
- **Scikit-learn** — Performance metrics and analytical tools  
- **Python** — Scripting and automation backbone  

---

## 🛠️ Development Workflow

1. **Data Preparation**  
   - Loaded MRI images from well-structured training and testing folders.  
   - Shuffled datasets and encoded labels dynamically.  
   - Applied brightness and contrast augmentations in real-time during training to boost robustness.

2. **Model Setup**  
   - Initialized VGG16 pretrained on ImageNet without its top layers.  
   - Froze most layers to preserve pretrained features; unfroze several top layers for targeted fine-tuning.  
   - Added custom classification layers with dropout and dense connections.

3. **Training**  
   - Used a custom data generator for efficient batch processing.  
   - Trained the model over multiple epochs with the Adam optimizer.  
   - Tracked and visualized accuracy and loss metrics throughout the training cycle.

4. **Evaluation**  
   - Assessed model performance with classification reports and confusion matrices on test data.  
   - Visualized results through heatmaps and ROC/AUC curves for detailed understanding.

5. **Inference**  
   - Preprocessed new MRI images and ran predictions with confidence scoring.  
   - Presented results alongside input images for interpretability.

---

## ☁️ Deployment

The system is deployed on **Render** to provide seamless web access. The backend is built using **Flask**, serving the model and handling image uploads smoothly. Explore the live demo here:

[https://mri-brain-tumor-diagnosis-system.onrender.com/](https://mri-brain-tumor-diagnosis-system.onrender.com/)

---

## 🎯 Conclusion

This MRI Brain Tumor Diagnosis System showcases how transfer learning with VGG16 can be effectively applied to challenging medical imaging tasks, even when working with limited data. By combining strategic data augmentation, fine-tuning, and comprehensive evaluation, the system achieves promising tumor classification accuracy.

That said, the model is **not 100% accurate** and is trained on a **relatively small dataset**, which may impact its performance on new, unseen cases. It should be viewed as a **supportive diagnostic aid** rather than a definitive medical tool.

This project lays a solid foundation for future AI-driven diagnostic applications and underscores deep learning’s potential to empower healthcare professionals with faster, more informed decision-making.

Going forward, expanding the dataset, including additional tumor classes, and progressing toward clinical deployment are exciting directions to explore.

---
