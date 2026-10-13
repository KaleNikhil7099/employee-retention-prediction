# Hugging Face Spaces configuration
---
title: RetentionAI Employee Retention Prediction
emoji: 🧬
colorFrom: teal
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# RetentionAI — Employee Retention Prediction Platform

> **Disclaimer:** This application uses the fictional IBM HR Analytics dataset. It does not contain records of real IBM employees.

A professional machine-learning web application for HR analytics and employee attrition prediction.

## Features

- 🔒 Secure authentication (register/login)
- 📦 IBM HR Analytics dataset explorer  
- ⚙️ Data preprocessing pipeline
- 🧪 Multi-model training (Logistic Regression, Random Forest, Gradient Boosting)
- 🔮 Single employee attrition prediction
- 📁 Batch CSV prediction with download
- 🤖 AI Chatbot powered by Hugging Face Mistral
- 📊 Feature importance and model insights
- 🐳 Docker-ready deployment

## Dataset Setup

1. Download from Kaggle: [IBM HR Analytics Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
2. Place `WA_Fn-UseC_-HR-Employee-Attrition.csv` in the `data/` directory.

## Local Setup

```bash
# 1. Clone this repository
git clone https://github.com/KaleNikhil39/employee-retention-prediction.git
cd employee-retention-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the Kaggle dataset to data/
# (see Dataset Setup above)

# 4. Set your Hugging Face API token (for the AI chatbot)
set HF_API_TOKEN=hf_your_token_here   # Windows
export HF_API_TOKEN=hf_your_token_here  # Linux/Mac

# 5. Run the app
streamlit run app.py
```

## Train the Model

```bash
python ml/train_pipeline.py
```

This trains Logistic Regression, Random Forest, and Gradient Boosting models, saves the best pipeline to `models/pipeline.pkl`, and stores metrics to `models/metrics.json`.

## Docker

```bash
# Build
docker build -t retention-ai .

# Run
docker run -p 7860:7860 -e HF_API_TOKEN=hf_... retention-ai

# Open
# http://localhost:7860
```

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Web Framework | Streamlit |
| ML | Scikit-learn (LR, RF, GBM) |
| Data | Pandas, NumPy |
| Visualization | Plotly |
| Auth DB | SQLite (PBKDF2 hashing) |
| AI Chatbot | Hugging Face Mistral API |
| Deployment | Docker + Hugging Face Spaces |

## License

MIT License. The IBM HR dataset is open-access from Kaggle for educational purposes.
Dataset license: [CC0 Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
