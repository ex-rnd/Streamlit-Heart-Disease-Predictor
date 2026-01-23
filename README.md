# ❤️ Heart Disease Predictor

A **Streamlit web app** that predicts the risk of heart disease using a trained **Logistic Regression model**.  
This project leverages patient health metrics to estimate whether an individual is at risk of heart disease, providing confidence scores and risk levels.

---

## ✨ Features
- 📊 Interactive UI built with **Streamlit**
- 🧠 Machine Learning model trained on **303 patient records**
- 🔍 Prediction outputs:
  - **Disease Risk / Healthy**
  - **Confidence score**
  - **Risk level (Low / Medium / High)**
- 📈 Probability visualization with **Plotly**
- 🧾 Input summary for transparency
- 🛡️ Educational tool (not a substitute for medical advice)

---

## 🛠️ Tech Stack
- **Python 3.8+**
- **Streamlit** – UI framework
- **Pandas / NumPy** – Data handling
- **Joblib** – Model persistence
- **Plotly** – Interactive charts
- **Scikit-learn** – Logistic Regression model

---

## 📂 Project Structure
```
heart-disease-predictor/
│── model/
│   └── heart_disease_model.pkl        # Trained ML model
│── notebook/
│   └── heart_scaler.pkl               # Feature scaler
│── app.py                              # Streamlit app
│── requirements.txt                    # Dependencies
│── README.md                           # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/heart-disease-predictor.git
cd heart-disease-predictor
```

2. **Create a virtual environment**
```
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. **Install dependencies**
```
pip install -r requirements.txt
```

4. **Run the app**
```
streamlit run app.py
```



















