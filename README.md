Real-Time Fraud Detection and Prevention System

An intelligent fraud detection framework that combines Artificial Intelligence (AI) and Multi-Criteria Decision-Making (MCDM) techniques to identify, analyze, and prevent fraudulent financial transactions in real time.

---

Live Demo

Try the application here:

https://mithlesh55-real-time-fraud-detection-system-fraud-v0xrju.streamlit.app


Project Overview

Financial fraud is becoming increasingly sophisticated in modern digital ecosystems. Traditional fraud detection methods often fail to provide transparent and explainable decisions.

This project integrates:

- Artificial Intelligence (Rule-Based Decision Logic)
- Analytical Hierarchy Process (AHP)
- Technique for Order Preference by Similarity to Ideal Solution (TOPSIS)
- Elimination and Choice Expressing Reality (ELECTRE)

to deliver accurate, explainable, and scalable fraud detection.

---

Objectives

- Detect suspicious transactions in real time
- Reduce false positives through multi-criteria evaluation
- Provide explainable fraud risk assessment
- Recommend effective fraud prevention strategies
- Support future integration with Machine Learning models

---

System Architecture

```text
┌─────────────────────────────┐
│ Transaction Input Layer     │
│ Amount, Frequency, IP, Time │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Data Preprocessing Module   │
│ Normalization & Validation  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ AI-Based Fraud Detection    │
│ Rule-Based Decision Engine  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ MCDM Risk Evaluation        │
│ AHP + TOPSIS + ELECTRE      │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Ensemble Decision Layer     │
│ Fraud / Safe Classification │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Fraud Prevention Module     │
│ Alerts & Recommendations    │
└─────────────────────────────┘
```

---

Technologies Used

### Programming Language
- Python

Libraries
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn

Decision-Making Techniques
- AHP
- TOPSIS
- ELECTRE

AI Approach
- Rule-Based Artificial Intelligence
- Ensemble Decision Logic

---

Project Structure

```text
Real-Time-Fraud-Detection-System
│
├── Fraud.py
├── mcdm_ai.py
├── Real_Time_Fraud_Detection_and_Prevention_using_mcdm.py
│
├── README.md
│
└── requirements.txt
```

---

Workflow

1. User enters transaction details
2. Data is preprocessed and normalized
3. AI rule engine evaluates fraud indicators
4. AHP calculates criteria importance
5. TOPSIS ranks transaction risk
6. ELECTRE performs outranking analysis
7. Ensemble model combines decisions
8. Final fraud status is generated
9. Prevention strategies are recommended

---

Key Features

Real-Time Fraud Detection
Monitors transactions instantly.

Risk Scoring
Assigns fraud risk scores using MCDM techniques.

Explainable Decision Making
Provides transparent and interpretable fraud decisions.

Ensemble Detection
Combines AHP, TOPSIS, and ELECTRE results.

Fraud Prevention Recommendations
Suggests security measures based on detected risks.

Interactive Dashboard
Built using Streamlit for visualization and analysis.



Application Screenshots

<img width="1920" height="4267" alt="image" src="https://github.com/user-attachments/assets/ef5879a1-ea9d-4721-b132-d10cfea4b4bc" />
<img width="1920" height="5293" alt="image" src="https://github.com/user-attachments/assets/f0dd46fc-1dde-46b0-bd6d-ee45feff39d1" />



Fraud Detection Criteria

The system evaluates transactions based on:

- Transaction Amount
- Transaction Frequency
- Transaction Time
- Location Change
- IP Address Risk
- Device Recognition

---

Installation

Clone the repository:

```bash
git clone https://github.com/MITHLESH55/Real-Time-Fraud-Detection-System.git
```

Move to project directory:

```bash
cd Real-Time-Fraud-Detection-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
streamlit run Fraud.py
```

---

Future Enhancements

- Machine Learning Integration
- Deep Learning Models
- Federated Learning
- Real Banking Dataset Integration
- Cloud Deployment
- Behavioral Biometrics
- Blockchain-based Fraud Prevention

---

Author

Mithlesh Yadav
B.Tech Computer Science & Engineering  
Symbiosis Institute of Technology, Pune, India

GitHub:
https://github.com/MITHLESH55

Linkedin:
https://www.linkedin.com/in/mithleshyadav977/

Live Demo: 
https://mithlesh55-real-time-fraud-detection-system-fraud-v0xrju.streamlit.app

---

Research Contribution

This project demonstrates a hybrid AI-MCDM framework for fraud detection, combining intelligent decision-making with explainable risk evaluation techniques suitable for modern financial systems.

---

If you found this project useful

Please consider giving it a ⭐ on GitHub.
