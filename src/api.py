# In src/api.py
import pandas as pd
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the entire pipeline object
pipeline = joblib.load('insurance_pipeline_v1.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Convert to DataFrame. The column names must match the original raw data
        # e.g., 'age', 'sex', 'bmi', 'smoker', etc.
        input_data = pd.DataFrame([data])
        
        # The pipeline handles everything: preprocessing AND prediction
        prediction = pipeline.predict(input_data)

        return jsonify({'predicted_charge': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)