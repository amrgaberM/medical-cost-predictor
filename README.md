# Medical Insurance Cost Predictor

A comprehensive, end-to-end machine learning application that predicts personal medical insurance costs. This project demonstrates a complete professional workflow, evolving from a foundational model built on clean data to an advanced model trained on complex, real-world datasets.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-2C2E3B?style=flat&logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)

## Overview

This project showcases advanced machine learning engineering practices through the development of two model versions, each addressing different data complexity levels and business requirements. The application demonstrates professional ML workflows including feature engineering, model experimentation, hyperparameter optimization, and production deployment.

## Key Features

### Advanced Feature Engineering
- **Custom Feature Creation**: Developed `procedure_category` features from raw medical data
- **Target Transformation**: Applied logarithmic transformation to handle extreme cost distribution skew
- **Data Quality Enhancement**: Comprehensive cleaning and preprocessing of real-world healthcare data

### Comprehensive Model Development
- **Systematic Model Selection**: Evaluated multiple algorithms (Lasso, Random Forest, XGBoost, LightGBM)
- **Scientific Comparison**: Rigorous performance evaluation using cross-validation and multiple metrics
- **Hyperparameter Optimization**: Implemented `GridSearchCV` for optimal model tuning

### Production-Ready Architecture
- **Pipeline Implementation**: Utilized `scikit-learn` pipelines to prevent data leakage and ensure reproducibility
- **Versioned Deployment**: RESTful API with versioned endpoints (`/v1`, `/v2`) following industry best practices
- **Scalable Design**: Modular structure supporting easy model updates and maintenance

## Project Structure

```
insurance-cost-predictor/
├── data/
│   └── raw/
│       ├── insurance.csv              # Phase 1: Clean Kaggle dataset
│       └── cms_data_raw.csv          # Phase 2: Real-world CMS data
├── notebooks/
│   ├── 1_phase_1_data_exploration.ipynb
│   ├── 2_phase_1_model_experimentation.ipynb
│   ├── 3_phase_1_final_pipeline.ipynb
│   ├── 4_phase_2_data_cleaning_and_eda.ipynb
│   ├── 5_phase_2_model_experimentation.ipynb
│   └── 6_phase_2_final_pipeline.ipynb
├── src/
│   └── api.py                        # Flask API implementation
├── models/
│   ├── insurance_pipeline_v1.joblib  # Phase 1 production model
│   └── insurance_pipeline_v2.joblib  # Phase 2 production model
├── requirements.txt
└── README.md
```

## Development Phases & Results

### Phase 1: Foundational Model ✅
**Objective**: Establish robust ML workflow with clean, structured data

- **Dataset**: Kaggle "Medical Cost Personal Datasets" (clean, preprocessed)
- **Approach**: Traditional feature engineering and model selection
- **Performance**: Mean Absolute Error (MAE) of ~$2,680
- **Key Learning**: Baseline workflow establishment and pipeline architecture

### Phase 2: Advanced Real-World Implementation ✅
**Objective**: Handle complex, messy real-world data challenges

- **Dataset**: CMS (Centers for Medicare & Medicaid Services) raw data
- **Challenges Addressed**:
  - Inconsistent data formats and missing values
  - High-dimensional categorical features
  - Extreme target variable skewness
- **Advanced Techniques**:
  - Custom feature engineering for medical procedure categorization
  - Log transformation for target variable normalization
  - Ensemble method optimization (LightGBM)
- **Performance**: MAE of ~$8,907, R² of 0.25
- **Achievement**: Realistic performance on complex healthcare data

## Technical Implementation

### Model Performance Comparison

| Phase | Dataset | Algorithm | MAE | R² | Data Complexity |
|-------|---------|-----------|-----|----|----|
| 1 | Kaggle Clean | Random Forest | ~$2,680 | 0.87 | Low |
| 2 | CMS Raw | LightGBM | ~$8,907 | 0.25 | High |

### API Endpoints

The application provides RESTful endpoints for both model versions:

- **Version 1**: `POST /predict/v1` - Clean data model
- **Version 2**: `POST /predict/v2` - Real-world data model

## Quick Start

### Prerequisites
- Python 3.9 or higher
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/insurance-cost-predictor.git
   cd insurance-cost-predictor
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the API server**
   ```bash
   python src/api.py
   ```

   The API will be available at `http://127.0.0.1:5000`

### Usage Examples

**Test Version 1 Model:**
```bash
curl -X POST http://127.0.0.1:5000/predict/v1 \
  -H "Content-Type: application/json" \
  -d '{"age": 25, "sex": "male", "bmi": 22.5, "children": 0, "smoker": "no", "region": "southeast"}'
```

**Test Version 2 Model:**
```bash
curl -X POST http://127.0.0.1:5000/predict/v2 \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "procedure_count": 3, "total_claim_amount": 15000}'
```

## Model Specifications

### Version 1 (v1)
- **Input Features**: age, sex, bmi, children, smoker, region
- **Target**: Insurance charges (direct prediction)
- **Algorithm**: Random Forest with optimized hyperparameters
- **Use Case**: Quick estimates with standard demographic data

### Version 2 (v2)
- **Input Features**: Enhanced feature set including procedure categories
- **Target**: Log-transformed insurance costs
- **Algorithm**: LightGBM with comprehensive hyperparameter tuning
- **Use Case**: Complex real-world predictions with detailed medical data

## Acknowledgments

- **Data Sources**: 
  - Kaggle Medical Cost Dataset (Phase 1 baseline)
  - [CMS Medicare Inpatient Hospitals Dataset](https://data.cms.gov/provider-summary-by-type-of-service/medicare-inpatient-hospitals/medicare-inpatient-hospitals-by-provider-and-service) (Phase 2 advanced implementation)
- **Technologies**: Scikit-learn, LightGBM, Flask, Python ecosystem
- **Methodology**: Industry-standard MLOps practices and model versioning
- **Performance Context**: Healthcare cost prediction R² benchmarks and real-world data modeling challenges
