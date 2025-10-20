# Machine Learning & Deep Learning Algorithms from Scratch

This repository contains implementations of core **machine learning** and **deep learning** algorithms built completely from scratch in **Python** and **NumPy**, with some neural network components using **TensorFlow** for comparison and experimentation.

---

## 📘 Overview

The project demonstrates how foundational ML and DL algorithms work under the hood without relying on high-level libraries like `scikit-learn` or `keras`.  
Each algorithm is written clearly and organized into logical modules for learning, experimentation, and extension.

---

## 🧠 Structure

```
algorithms from scratch/
│
├── ML algorithms/           # Classical ML implementations (NumPy-based)
│   ├── LinearRegression.ipynb
│   ├── LogisticRegression.ipynb
│   ├── DecisionTree.ipynb
│   ├── KMeans.ipynb
│   └── PCA.ipynb
│
├── DL algorithms/           # Neural networks & deep learning models
│   ├── backpropagation.ipynb
│   ├── cnn_model.ipynb
│   ├── rnn_model.ipynb
│   ├── transformer.ipynb
│   ├── autoencoders.ipynb
│   └── sequential_model_tf.ipynb
│
├── losses/                  # Loss functions (MSE, Cross-Entropy, etc.)
├── metrics/                 # Evaluation metrics (Accuracy, Confusion Matrix, etc.)
├── optimizers/              # Optimizers (SGD, Adam, RMSProp, etc.)
├── preprocessing/           # Data preprocessing utilities
└── utils/                   # Helper functions
```

---

## ⚙️ Tech Stack

- **Python 3.10+**
- **NumPy** — mathematical and array operations  
- **TensorFlow** — neural network models  
- **Jupyter Notebook** — interactive implementation and visualization  

---

## 🚀 Features

- From-scratch ML algorithms (no external ML libraries)  
- Full neural network pipeline built manually (forward, backward, gradient)  
- Deep learning models using TensorFlow for benchmarking  
- Well-structured and modular code  
- Educational clarity for ML fundamentals  

---

## 🧩 Example Topics

- Linear & Logistic Regression  
- Decision Trees  
- K-Means Clustering  
- Principal Component Analysis (PCA)  
- Feedforward Neural Networks  
- CNNs, RNNs, Transformers  
- Loss and Metric functions  
- Optimizers and Gradient Computation  

---

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ramcharan2905/algorithms-from-scratch.git
   cd algorithms-from-scratch
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
   Open any notebook under `ML algorithms/` or `DL algorithms/` to explore and run the implementations.

---

## 📈 Learning Purpose

This repository is designed for students and practitioners who want to:

- Understand the math and logic behind ML/DL models  
- Build end-to-end algorithms from basic principles  
- Visualize and test implementations step by step  

---

## 👤 Author

**Gudala Geeta Ramcharan**  
Machine Learning Enthusiast | Python Developer  
GitHub: [Ramcharan2905](https://github.com/Ramcharan2905)

---

## ⭐ Acknowledgments

Inspired by core ML theory, Andrew Ng’s ML course, and TensorFlow educational resources.
