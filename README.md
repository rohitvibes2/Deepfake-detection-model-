#DeepFake Image Detection using MesoNet (Meso4)

This project implements a DeepFake image detection system using a Convolutional Neural Network (CNN) based on the MesoNet (Meso4) architecture. The goal is to classify facial images as Real or Fake (DeepFake) by learning visual inconsistencies and artifacts introduced during DeepFake generation.

The system provides a complete pipeline including:

Data preprocessing

Model training

Model evaluation

Single-image inference with confidence score

This project is intended for educational and academic purposes, particularly for understanding how deep learning can be applied to media forensics.

 ##Objectives

Detect DeepFake images using a lightweight CNN architecture

Train a model on labeled real and fake face images

Evaluate performance on unseen test data

Predict the authenticity of a single image with a confidence score

Build a reusable and modular DeepFake detection pipeline

# Model Architecture

The project uses Meso4, a compact CNN specifically designed for DeepFake detection.

Key architectural components:

Convolutional layers with varying kernel sizes

Batch normalization for stable training

Max pooling layers for spatial down-sampling

Dropout layers to reduce overfitting

Fully connected layers for classification

Sigmoid output layer for binary classification

Output Interpretation:

Output value ∈ [0, 1]

> 0.5 → Real

≤ 0.5 → Fake

📂 Dataset Structure

The dataset must be organized in the following directory structure:

data/
├── train/
│   ├── real/
│   └── fake/
├── test/
│   ├── real/
│   └── fake/


Images placed in real/ are treated as Real

Images placed in fake/ are treated as DeepFake

Training and testing datasets are kept strictly separate

### Note:
Due to size and licensing restrictions, the dataset is not included in this repository. Users must prepare their own dataset following the above structure.

⚙️ Data Preprocessing

All images are resized to 256 × 256

Pixel values are normalized to the range [0, 1]

Data is loaded using Keras ImageDataGenerator for memory-efficient training

🏋️ Model Training

Loss Function: Binary Cross-Entropy

Optimizer: Adam

Training performed using batches of images

Validation performed on the test dataset

Trained model weights are saved to avoid retraining and to enable fast inference.

🔍 Inference & Prediction

The trained model can be used to:

Predict authenticity of test images

Perform single-image prediction by providing an image path

Example output:
Prediction: FAKE
Confidence: 0.32


The confidence score represents the model’s estimated probability that the image is real.

# Technologies Used

Python

TensorFlow / Keras

NumPy

Matplotlib

PIL / OpenCV

Jupyter Notebook

# Limitations

MesoNet is a lightweight model and may struggle with high-quality or unseen DeepFake techniques

Performance depends heavily on dataset size and diversity

Not intended for production-level DeepFake detection

Best suited for learning, experimentation, and academic projects

# Conclusion

This project demonstrates a practical application of deep learning in the field of DeepFake image detection. It provides insight into how CNNs can be trained to identify manipulated images while also highlighting the challenges involved in detecting modern DeepFake content.

📄 License

This project is licensed under the MIT License.
