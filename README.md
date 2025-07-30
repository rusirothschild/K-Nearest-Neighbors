# 📍 K-Nearest Neighbors (KNN) from Scratch

This project implements the **K-Nearest Neighbors (KNN)** classification algorithm entirely from scratch using Python and NumPy — no external machine learning libraries. It demonstrates a hands-on understanding of how distance-based models work internally.

## 💡 What It Does

- Calculates distances between input data and training examples
- Identifies the `k` nearest neighbors for each query
- Performs majority vote for classification
- Supports both Euclidean and Manhattan distances

## 🧠 Skills Demonstrated

- Custom ML model development with Python classes
- Vectorized distance calculations with NumPy
- Manual prediction loop without `sklearn`
- Reinforcement of data structures and control logic


## 🚀 How to Use

```python
from knn import KNN

model = KNN(k=3, distance_metric="euclidean")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
