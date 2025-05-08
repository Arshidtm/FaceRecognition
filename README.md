# FaceRecognition

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)

A production-ready face recognition system with web interface built with FastAPI, MTCNN, and FaceNet.

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# Clone repository
https://github.com/Arshidtm/FaceRecognition.git
cd FaceRecognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# run on local
uvicorn main:app --reload

# acsess the api
https://13.48.28.54/
(shows not secure because not using a domain name)

