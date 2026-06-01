# Retinopathy Detection Project

Detect diabetic retinopathy in retinal scans using deep learning (EfficientNetB0).

## 🚀 How to Run

### 1. Prerequisite Setup (First Time Only)
Open a terminal (e.g. PowerShell or Command Prompt) and navigate to this folder.
- Create a virtual environment:
  ```bash
  python -m venv venv
  ```
- Activate it:
  - Windows: `.\venv\Scripts\activate`
  - Mac/Linux: `source venv/bin/activate`
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Evaluate Your Model (Metrics)
To quickly see the performance of your current model (Accuracy, Confusion Matrix, etc.) without retraining:
```bash
python evaluate_model.py
```

### 3. Run the Web Application
To start the retinopathy detection web interface:
```bash
python app.py
```
After it starts, open your browser and go to: `http://localhost:5000`

### 4. Train the Model (Optional)
If you update your dataset and need to retrain the model from scratch:
```bash
python train_model.py
```

## 📂 Data Structure
Ensure your dataset is organized as follows:
```text
dataset/
├── train/
│   ├── No_DR/
│   └── DR/
└── test/
    ├── No_DR/
    └── DR/
```
