# Industrial Defect Detection System with OpenCV, AI & Cloud Integration

## 📌 Overview
A **Streamlit-powered web application** that leverages **OpenCV** for advanced computer vision tasks, enhanced with **AI-driven defect detection** and designed for **industrial quality control**. Integrated with cloud storage options for **scalability and secure storage** of results and logs.

This system serves educational, research, and industrial purposes, enabling image-based inspection, **surface analysis**, and **dimensional measurement** with real-time AI insights.

---

## 🚀 Features
- ✅ Face and eye detection via Haar Cascades
- ✅ Surface defect detection (Cracks, Scratches, Spots, Line Defects)
- ✅ Dimensional measurement using pixel-to-mm calibration
- ✅ Surface texture and roughness analysis
- ✅ Cloud integration for storing images and logs
- ✅ Downloadable processed images
- ✅ Real-time image annotations and metadata

---

## 🛠️ System Architecture

### 📌 Frontend
- **Framework:** Streamlit SPA (Single Page App)
- **UI:** Sidebar controls for selecting operations
- **State Management:** Streamlit Session State

### 📌 Backend
- **Core Engine:** OpenCV (cv2), Numpy, Pillow
- **AI Model Support:** Pre-trained Haar cascades + placeholders for custom ML models
- **Cloud Storage:** (Optionally) Integrate AWS S3, Azure Blob Storage, or Google Cloud Storage
- **Logging:** Captures metadata and actions into cloud or local storage

---

## 🔄 Data Flow

1. 📤 **Upload Image**
2. ⚙️ **Select Processing Type (Face Detection, Defect Analysis, Measurement)**
3. 🔍 **Apply Processing with OpenCV & AI Models**
4. 🎯 **Display Processed Results Side by Side**
5. ☁️ **Optionally Save Results to Cloud**
6. 📥 **Download Processed Outputs**

---

## 🗂️ Project Structure
├── app.py # Streamlit UI & main control logic
├── cv_processor.py # Core computer vision logic
├── utils.py # Utilities: file handling, conversions
├── models/ # Haar cascades and other models
│ ├── haarcascade_frontalface_default.xml
│ └── haarcascade_eye.xml
├── cloud/ # Optional: Cloud integration utilities
│ ├── s3_utils.py # Example for AWS S3 operations
└── requirements.txt # Dependencies

yaml
Copy
Edit

---

## ⚡ Installation & Running Locally
1. **Clone Repo**
```bash
git clone https://github.com/birukG09/Industrial-Defect-Detection-System.git
cd Industrial-Defect-Detection-System
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the App

bash
Copy
Edit
streamlit run app.py
🧩 Key Components
✅ CVProcessor Class
Handles:

Face & eye detection

Edge detection

Surface defect detection

Dimensional analysis

AI-based texture insights

✅ Cloud Utilities (Optional)
To:

Upload processed images and logs to AWS S3 or any cloud provider

Retrieve logs for audits

✅ Utilities
Download model files from the web

Image format conversions (PIL <-> OpenCV)

State management helpers

🔒 Security & Production Considerations
Validate file uploads (formats, size limits)

Cache or persist models on server for efficiency

Monitor cloud API keys & access

Deploy on a VM or cloud service with GPU for intensive processing

✅ Deployment Options
Local: Python + Streamlit (best for demos)

Dockerized Deployment: Containerize for production

Cloud VM: Deploy on AWS, Azure, GCP

Serverless Extensions: Add APIs via FastAPI for integrations

🛠️ External Dependencies
streamlit

opencv-python

numpy

pillow

boto3 (optional for AWS S3)

🔮 Future Enhancements
✅ Integrate Deep Learning Models (YOLO, SSD) for real-time detection

✅ Build REST APIs using FastAPI for scalable backend

✅ Add database logging (MongoDB/PostgreSQL) for industrial logs

✅ Build mobile-friendly UI with Streamlit Components

🧑‍💻 Author
Biruk G | GitHub Profile

yaml
Copy
Edit
