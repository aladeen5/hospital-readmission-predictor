Model Description
    - Model used: Logistic Regression
    - Why: It is interpretable, efficient, and suitable for binary outcomes (readmitted vs not readmitted).
    - Features: Age, gender, blood pressure, BMI, cholesterol, diabetes, hypertension, medication count, length of stay, discharge destination, etc.
    - Output:
         High risk (probability ? 0.5)
         Low risk (probability < 0.5)
Preprocessing Pipeline
    9. Handle missing values and outliers
    10. Encode categorical variables (e.g., gender, diabetes status)
    11. Normalize numerical features
    12. Feature engineering:
         Age grouped into ranges
         Derived features like BMI categories

Installation and Setup
1. Clone the project
git clone https://github.com/aladeen5/hospital-readmission-predictor.git
2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)
3. Install dependencies
pip install -r requirements.txt
4. Run the app
uvicorn app:app --reload
5. Access the dashboard
    - Open: http://127.0.0.1:8000/dashboard
    - API Docs: http://127.0.0.1:8000/docs

             
Example API Request
Endpoint: /predict_readmission
Method: POST
Body (JSON):
{
  "age": 55,
  "gender": "Female",
  "blood_pressure": "120/80",
  "cholesterol": 210,
  "bmi": 27.5,
  "diabetes": "Yes",
  "hypertension": "No",
  "medication_count": 3,
  "length_of_stay": 5,
  "discharge_destination": "Home"
}

Response:
{
  "prediction": "High risk",
  "readmission_probability": 0.78
}
