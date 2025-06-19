# Image-captioning
# 🖼️ Image Captioning Web App

This project is a **Flask-based web application** that automatically generates **descriptive captions** for uploaded images using state-of-the-art deep learning models like **BLIP** and **ViT-GPT2**. It also includes fallback support for Hugging Face and Replicate APIs.

🔍 Features

- Upload an image and get one or more AI-generated captions.
- Switch between BLIP and ViT-GPT2 models.
- Automatic fallback to Hugging Face Inference API if local models aren't available.
- RESTful API endpoints with JSON responses.
- Modern, minimal, user-friendly frontend.

🛠️ Technologies Used

- **Python**, **Flask**, **PIL**, **Torch**, **Transformers**
- **Hugging Face models**: `Salesforce/blip-image-captioning-base`, `nlpconnect/vit-gpt2-image-captioning`
- **HTML/CSS/JS** for frontend
- Optional integration: **Replicate API** and **Hugging Face API**
