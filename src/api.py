import pandas as pd
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# --- Load both model pipelines at startup ---
# This dictionary will hold your trained models.
pipelines = {}

try:
    pipelines['v1'] = joblib.load('insurance_pipeline_v1.joblib')
    pipelines['v2'] = joblib.load('insurance_pipeline_v2.joblib')
    print("Pipelines v1 and v2 loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading models: Could not find {e}. Make sure both .joblib files are in the root directory.")
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")

# =============================================================================
# API Endpoint for the V1 Model (Simple Kaggle Data)
# =============================================================================
@app.route('/predict/v1', methods=['POST'])
def predict_v1():
    """Handles requests for the simple, Phase 1 model."""
    if 'v1' not in pipelines:
        return jsonify({'error': 'Model v1 is not available.'}), 500

    try:
        data = request.get_json()
        
        # Expected input: {'age': 30, 'sex': 'male', 'bmi': 25.0, ...}
        input_data = pd.DataFrame([data])
        
        # The pipeline handles preprocessing and prediction
        prediction = pipelines['v1'].predict(input_data)

        # The V1 model predicts cost directly (no log transform)
        final_prediction = prediction[0]

        return jsonify({
            'predicted_charge': final_prediction,
            'model_version': 'v1'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# =============================================================================
# API Endpoint for the V2 Model (Advanced CMS Data)
# =============================================================================
@app.route('/predict/v2', methods=['POST'])
def predict_v2():
    """Handles requests for the advanced, Phase 2 model."""
    if 'v2' not in pipelines:
        return jsonify({'error': 'Model v2 is not available.'}), 500

    try:
        data = request.get_json()
        
        # Expected input: {'state_abbr': 'CA', 'ruca_code': 1.0, 'total_discharges': 50, ...}
        input_data = pd.DataFrame([data])
        
        # The pipeline handles preprocessing and prediction
        prediction_log = pipelines['v2'].predict(input_data)
        
        # --- IMPORTANT ---
        # Convert the prediction from the log scale back to dollars
        final_prediction = np.expm1(prediction_log[0])

        return jsonify({
            'predicted_charge': final_prediction,
            'model_version': 'v2'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# =============================================================================
# Root URL - To confirm the API is running
# =============================================================================
@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Insurance Cost Predictor API is running.', 'available_models': list(pipelines.keys())})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
