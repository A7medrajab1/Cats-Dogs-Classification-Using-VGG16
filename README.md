# 🐶🐱 Dog vs Cat Image Classification using VGG16

This project builds a deep learning model based on the VGG16 architecture to classify images as either **dogs** or **cats**. The model is trained on the [Kaggle PetImages dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset).

---

## 📁 Dataset

- **Source**: Kaggle `PetImages` dataset
- **Structure**:
  ```
  PetImages/
  ├── Cat/
  └── Dog/
  ```

- Each image is resized to **224×224** to match VGG16 input requirements.

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV (for image loading)
- Matplotlib & Seaborn (for plotting)
- Scikit-learn (for evaluation metrics)
- NumPy, tqdm

---

## 📦 Model Architecture

- Preprocessing:
  - Resize all images to `224x224`
  - Normalize pixel values to `[0, 1]`
  - Limit number of images per class (for Kaggle memory constraints)
- Model:
  - Pre-trained **VGG16 (convolutional base only)**
  - Custom classification head with:
    - Flatten layer
    - Dense + ReLU
    - Dropout
    - Final Dense (1 neuron + sigmoid)
- Training:
  - `batch_size = 32`
  - `epochs = 12`
  - `validation_split = 0.1`
  - `train_test_split` used to separate 10% of data for testing

---

## 📊 Evaluation

After training, the model is evaluated using:

- Accuracy and loss (train, validation, test)
- Confusion matrix
- Classification report (precision, recall, F1-score)
- ROC curve

### 🔍 Sample Metrics

| Metric         | Value    |
|----------------|----------|
| Test Accuracy  | 95.67%    |
| F1-Score       | ~0.96    |

---

## 📈 Visualizations

- Accuracy & loss curves
- Confusion matrix heatmap
- ROC Curve

---

## ✅ How to Run

1. Clone this repository or run in [Kaggle Notebooks](https://www.kaggle.com/).
2. Ensure the dataset is correctly placed under `PetImages/Dog` and `PetImages/Cat`.
3. Run the notebook step-by-step:
   - Load & preprocess images
   - Build and train the model
   - Evaluate and visualize performance

---

## 🧠 Future Improvements

- Use `ImageDataGenerator` or `tf.data` to handle large datasets
- Apply real-time data augmentation
- Experiment with MobileNet or EfficientNet for faster training
- Add model checkpointing and early stopping

---

## 📌 Author

- Ahmed Ragab  
- Powered by Kaggle, TensorFlow, and your GPU ❤️

---