# 🤟 Real-Time Arabic Sign Language Alphabet Recognition

A deep learning and computer vision project for recognizing **Arabic Sign Language alphabet signs in real time** using **MobileNetV2, TensorFlow, and OpenCV**.

The system classifies **31 Arabic sign classes** from RGB images and can use a webcam to display the predicted sign together with the model's confidence score.

---

## 📌 Project Overview

This project was developed to build an AI-based system capable of recognizing Arabic Sign Language alphabet signs from images and deploying the trained model in a real-time webcam application.

The project applies several Artificial Intelligence and Machine Learning concepts, including:

- Deep Learning
- Computer Vision
- Convolutional Neural Networks
- Transfer Learning
- Fine-Tuning
- Data Preprocessing
- Data Augmentation
- Label Encoding
- Multiclass Classification
- Model Evaluation
- Ensemble Learning
- Real-Time Inference using OpenCV

The final system receives a hand sign through a camera, processes the image, passes it through a trained MobileNetV2 model, and displays the predicted Arabic sign together with a confidence score.

---

# 🎯 Project Objectives

- Develop an AI-based system capable of recognizing Arabic Sign Language alphabet signs.
- Classify **31 different Arabic sign classes**.
- Apply deep learning and computer vision techniques to hand-sign recognition.
- Use **MobileNetV2 transfer learning** instead of training a CNN completely from scratch.
- Improve model performance using **fine-tuning and data augmentation**.
- Evaluate the model using accuracy, precision, recall, F1-score, confusion matrices, and per-class performance.
- Integrate the trained model with **OpenCV** for real-time webcam recognition.
- Build a foundation for a future **sign-to-speech communication system**.

---

# 🔮 Future Vision

The current system recognizes individual Arabic alphabet signs.

A future version could extend the project to recognize:

- Complete words
- Sequences of signs
- Full sentences
- Continuous sign language

The system could also be connected to a **text-to-speech engine**, allowing recognized signs to be converted into audible speech.

A possible future pipeline could be:

```text
Hand Signs
    ↓
Camera
    ↓
AI Sign Recognition
    ↓
Letters / Words
    ↓
Sentence Construction
    ↓
Text-to-Speech
    ↓
🔊 Spoken Output
