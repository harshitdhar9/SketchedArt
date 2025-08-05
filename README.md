# 🖼️ SketchedArt

> **Recognize hand-drawn sketches** with a lightweight CNN model. Featuring a modern Gradio interface for testing and a FastAPI backend for integration. Built with PyTorch.

---

## 📌 Features

- 🧠 CNN-based sketch recognition using PyTorch
- ⚡ FastAPI REST API for backend use
- 🎛️ Gradio UI for real-time sketch testing
- 🐍 Lightweight, CPU-friendly environment
- 🖥️ Run locally in VS Code with ease

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone git@github.com:harshitdhar9/SketchedArt.git
cd SketchedArt
```

### 2. Create & Activate Virtual Environment
```bash
#On mac/linux
python3 -m venv venv
source venv/bin/activate

#On windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Gradio UI
```bash
cd gradio_ui
python app.py
```

### 5.FastAPI Backend
```bash
cd backend
uvicorn app:app --reload
```
