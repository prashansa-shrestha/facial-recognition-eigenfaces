# Facial Recognition using Eigenfaces and kNN

## Overview

Facial Recognition is crucial in modern-day security-systems, attendance tracking and healthcare.
This project demonstrates how machine learning and linear algebra concepts can be used together for image-based recognition system.

This algorithm uses eigenfaces and k-Nearest Neighbors classification, along with dimesnionality reduction for increased efficiency.

---

## 🧩 How It Works

1. **Image Loading**
   - All facial images are loaded from the `lfw/` dataset directory using the `load_image()` function from `load_images.py`.
   - Each image is resized and converted into a flattened numerical array.

2. **Mean Face Calculation**
   - The average face is computed across all images.
   - Each image is then centered by subtracting this mean face.

3. **Covariance Matrix & Eigenfaces**
   - A covariance matrix is built from the centered data.
   - The eigenvalues and eigenvectors of this matrix are computed.
   - The top `k` eigenvectors correspond to the most significant features called the Eigenfaces.

4. **Projection**
   - Each centered image is projected into the new eigenspace, representing it by fewer features.

5. **Classification (kNN)**
   - The projected images are split into training and testing sets.
   - A kNN classifier is used to predict labels for the test images.
   - The accuracy of the model is printed.

6. **Visualization**
   - The top 10 Eigenfaces are displayed using Matplotlib.

---

## ⚙️ Requirements

Ensure you have **Python 3.8+** installed.  
Install dependencies using pip:

```bash
pip install numpy opencv-python matplotlib scikit-learn
````

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/facial-recognition-eigenfaces.git
cd facial-recognition-eigenfaces
```

### 2. Prepare the dataset

Download the LFW Deep Funneled dataset from [here](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) and export into:

```
lfw/lfw-deepfunneled/
```

Your folder structure should look like:

```
facial-recognition-eigenfaces/
│
├── faces.py
├── knn.py
├── load_images.py
├── faces.ipynb
├── lfw/
│   └── lfw-deepfunneled/
│       └── [person folders and images]
```

### 3. Run the program

```bash
run the main.ipynb file
```

### 4. View Results

* 

---

## 📊 Example Output

```
Loading images...
Loaded 13233 images of 5749 labels (people).
Centering face
Computing covariance matrix
Computing eigenvalues and eigenvectors
100 Eigenvectors identified
Accuracy:  0.88
```

And the visualization shows:

* Mean Face
* Top 10 Eigenfaces
* Recognition accuracy

*(You can add screenshots later like below)*

```markdown
![Mean Face](images/mean_face.png)
![Eigenfaces](images/eigenfaces.png)
![Results](images/results.png)
```

---

## 🧠 Key Concepts

| Concept                                | Description                                                                 |
| -------------------------------------- | --------------------------------------------------------------------------- |
| **PCA (Principal Component Analysis)** | Reduces high-dimensional image data while preserving most variance          |
| **Eigenfaces**                         | Principal components representing facial features                           |
| **Covariance Matrix**                  | Measures how pixel values vary together                                     |
| **kNN (k-Nearest Neighbors)**          | Classifies faces by comparing their eigenspace projections                  |
| **LFW Dataset**                        | Labeled Faces in the Wild – a standard benchmark for face recognition tasks |

---

## 📚 References

* [Labeled Faces in the Wild (LFW Dataset)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
* [Eigenfaces for Recognition — Turk & Pentland, 1991](https://ieeexplore.ieee.org/document/139758)

---

## 👩‍💻 Author

**Prashansa Shrestha**
Pulchowk Campus, Tribhuvan University
079bct061.prashansa@pcampus.edu.np
