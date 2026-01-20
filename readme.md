# Loan Approval Prediction System

This project is an end-to-end Machine Learning system for predicting loan approval status using applicant information.  
The best-performing Random Forest model achieved **99.55% accuracy** and the system is deployed using **Gradio**.

---

## Project Overview

The goal of this project is to automate loan approval decisions using machine learning.  
It uses demographic, financial, and asset-related features to predict whether a loan application will be approved or rejected.

---

## Model Performance

Best Model: Random Forest  
Accuracy: 99.55%

Evaluation metrics used:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## Dataset Features

- no_of_dependents  
- education  
- self_employed  
- income_annum  
- loan_amount  
- loan_term  
- cibil_score  
- residential_assets_value  
- commercial_assets_value  
- luxury_assets_value  
- bank_asset_value  

### Engineered Features
- monthly_income  
- monthly_loan_payment  
- total_assets  
- debt_to_income_ratio  
- asset_to_loan_ratio  

---

## Project Structure

ML Final Model Deployment/
│
├── app.py
├── train.py
├── loan_approval_model.pkl
├── requirements.txt
├── README.md
└── env/
## Technologies Used

- Python
- scikit-learn
- XGBoost
- Pandas
- NumPy
- Gradio
- Joblib

---

## How to Run the Project

1. Activate the virtual environment:
env\Scripts\activate

markdown
Copy code

2. Install dependencies:
pip install -r requirements.txt

markdown
Copy code

3. Run the application:
python app.py

css
Copy code

4. Open the browser and go to:
http://127.0.0.1:7360

yaml
Copy code

---

## Application

The Gradio web interface allows users to input applicant details and receive:
- Loan approval status
- Approval probability

---

## Future Work

- Model explainability
- Bias and fairness analysis
- Cloud deployment
- Performance optimization

---

## Author

Sharmin Akter Liza
Data Science and Machine Learning Enthusiast
