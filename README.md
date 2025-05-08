# FaceRecognition

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![FaceNet](https://img.shields.io/badge/FaceNet-5C3EE8?style=for-the-badge&logo=python&logoColor=white)
![MTCNN](https://img.shields.io/badge/MTCNN-FF6D01?style=for-the-badge&logo=opencv&logoColor=white)

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
streamlit run app.py

# acsess the api
https://13.48.28.54/
(shows not secure because not using a domain name)

