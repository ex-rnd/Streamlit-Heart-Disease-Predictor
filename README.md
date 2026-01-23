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

---

## 📊 Usage
1. Launch the app in your browser (default: http://localhost:8501).

2. Enter patient details:
```
- Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol, etc.
```

4. Click `Predict Heart Disease`.

5. View:
```
- Prediction result (Disease Risk / Healthy)
- Confidence score
- Risk level
- Probability chart
- Input summary
```

---

## 🧠 Model Information
- Algorithm: Logistic Regression
- Training Dataset: 303 patients
- Features: 13 attributes (age, sex, chest pain type, cholesterol, max heart rate, etc.)
- Accuracy: ~85%

---

## ⚠️ Disclaimer
- This project is for educational purposes only.
- It is not a substitute for professional medical advice, diagnosis, or treatment.
- Always consult a qualified healthcare provider for medical concerns.

---

## 🤝 Contributing
- Contributions are welcome!
1. Fork the repo
2. Create a feature branch (feature/new-ui)
3. Commit changes
4. Open a Pull Request

---

## 📜 License
- This project is licensed under the MIT License – feel free to use and modify.
