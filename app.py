import gradio as gr
import joblib
import pandas as pd
model = joblib.load("loan_approval_model.pkl")
education_map = {
    "Graduate": 1,
    "Not Graduate": 0
}

self_employed_map = {
    "Yes": 1,
    "No": 0
}

def predict_loan(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
):
    try:
        education = education_map[education]
        self_employed = self_employed_map[self_employed]
        monthly_income = income_annum / 12 if income_annum > 0 else 0
        monthly_loan_payment = loan_amount / loan_term if loan_term > 0 else 0

        total_assets = (
            residential_assets_value +
            commercial_assets_value +
            luxury_assets_value +
            bank_asset_value
        )

        debt_to_income_ratio = (
            monthly_loan_payment / monthly_income if monthly_income > 0 else 0
        )

        asset_to_loan_ratio = (
            total_assets / loan_amount if loan_amount > 0 else 0
        )
        data = pd.DataFrame([{
            "no_of_dependents": no_of_dependents,
            "education": education,
            "self_employed": self_employed,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets_value,
            "commercial_assets_value": commercial_assets_value,
            "luxury_assets_value": luxury_assets_value,
            "bank_asset_value": bank_asset_value
        }])

        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]

        status = "Loan Approved" if prediction == 1 else " Loan Rejected"
        return status, round(probability, 4)

    except Exception as e:
        return f"Error: {e}", 0
inputs = [
    gr.Number(label="Number of Dependents"),
    gr.Dropdown(["Graduate", "Not Graduate"], label="Education"),
    gr.Dropdown(["Yes", "No"], label="Self Employed"),
    gr.Number(label="Annual Income"),
    gr.Number(label="Loan Amount"),
    gr.Number(label="Loan Term (Months)"),
    gr.Number(label="CIBIL Score"),
    gr.Number(label="Residential Assets Value"),
    gr.Number(label="Commercial Assets Value"),
    gr.Number(label="Luxury Assets Value"),
    gr.Number(label="Bank Asset Value")
]

outputs = [
    gr.Text(label="Loan Status"),
    gr.Number(label="Approval Probability")
]

iface = gr.Interface(
    fn=predict_loan,
    inputs=inputs,
    outputs=outputs,
    title=" Loan Approval Prediction System",
    description="Enter applicant details to predict loan approval status."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7360)
