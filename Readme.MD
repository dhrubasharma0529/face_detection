# 🧠 Face Recognition System using DeepFace and SVM

This project is a **Real-Time Face Recognition System** built with **OpenCV**, **DeepFace**, and **Scikit-learn (SVM)**.  
It captures faces from a webcam, generates embeddings using **DeepFace’s ArcFace model**, and classifies them with a **pre-trained SVM model**.

---

## 🚀 Features

- **Real-time face detection** using OpenCV’s Haarcascade classifier  
- **Face embedding extraction** using DeepFace (ArcFace model)  
- **SVM-based face classification** for recognition  
- **Multi-threaded prediction** to keep video streaming smooth  
- **Live labeling** of recognized faces on the camera feed  
- **Model loading via Joblib** (scaler + trained SVM)  

---

## 🧩 Tech Stack

| Component | Description |
|------------|-------------|
| **Python** | Main programming language |
| **OpenCV** | For real-time video capture and face detection |
| **DeepFace** | For generating facial embeddings using ArcFace |
| **Scikit-learn** | For SVM model training and scaling embeddings |
| **Joblib** | To save and load trained models efficiently |
| **Threading** | For parallel inference to maintain performance |