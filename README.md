# Medical Insurance Cost Predictor

A complete, end-to-end machine learning application that predicts personal medical insurance costs. This project demonstrates a full professional workflow, starting with a foundational model built on a clean dataset and evolving to a more advanced model trained on complex, real-world data.

## Key Features

  - **Model Experimentation:** Systematically auditioned multiple regression models (Linear Regression, Random Forest, SVR, Gradient Boosting) to identify the best performer.
  - **Hyperparameter Tuning:** Used `GridSearchCV` to fine-tune the champion model (Gradient Boosting) for optimal performance.
  - **Robust Pipeline Architecture:** Implemented `scikit-learn` Pipelines to bundle preprocessing and modeling steps, eliminating training-serving skew and creating a professional, maintainable system.
  - **Live API:** Deployed the final model pipeline via a Flask API, making it available for live predictions.
  - **Professional Workflow:** Followed a two-phase development plan, proving the concept with a clean dataset before tackling a more complex, real-world dataset.

-----

## Project Structure

The repository is organized to separate concerns, making the project clean and easy to navigate.

```
insurance-cost-predictor/
├── data/
│   └── raw/
│       ├── insurance.csv         (Phase 1 Data)
│       └── cms_data_raw.csv      (Phase 2 Data - Example)
├── notebooks/
│   ├── phase_1_exploration.ipynb
│   ├── phase_1_experimentation.ipynb
│   └── phase_1_final_pipeline.ipynb
├── src/
│   └── api.py
├── insurance_pipeline_v1.joblib
└── README.md
```

-----

## Workflow & Methodology

This project is built in two distinct phases to demonstrate a comprehensive skill set.

### Phase 1: Foundational Model (Complete) ✅

The initial phase involved building a complete, end-to-end application using the clean Kaggle "Medical Cost Personal Datasets". This phase focused on establishing a robust workflow:

1.  **Data Analysis:** Gained key insights from the data.
2.  **Model Audition:** Scientifically tested multiple algorithms to select **Gradient Boosting** as the champion model.
3.  **Pipeline Creation:** Built a `scikit-learn` Pipeline to ensure preprocessing was robust and reproducible.
4.  **API Deployment:** Served the final pipeline through a live Flask API.

### Phase 2: Advanced Model with Real-World Data (In Progress) 🔬

This phase demonstrates the ability to handle the challenges of messy, real-world data. The goal is to build a new, more powerful model using a complex dataset from a source like **CMS (Centers for Medicare & Medicaid Services)**.

This phase will involve:

  - **Advanced Data Cleaning:** Handling missing values, correcting data types, and resolving inconsistencies in a large, raw dataset.
  - **Feature Engineering:** Creating new, impactful features from the complex data to improve model performance.
  - **Building a New Pipeline:** Constructing a new preprocessing and modeling pipeline tailored to the complexities of the new dataset.
  - **Comparative Analysis:** Comparing the performance of the advanced model against the foundational model.

-----

## How to Run This Project

Follow these steps to run the **Phase 1** application on your local machine.

**1. Clone the repository:**

```bash
git clone https://github.com/YourUsername/insurance-cost-predictor.git
cd insurance-cost-predictor
```

**2. Create and activate a virtual environment (recommended):**

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Run the API server:**
To start the server with the Phase 1 model, run:

```bash
python src/api.py
```

The API will be running at `http://127.0.0.1:5000`.

**5. Test the API:**
You can send a `POST` request to the `/predict` endpoint to get a live prediction. Here is an example using PowerShell:

```powershell
$jsonData = '{"age": 52, "sex": "female", "bmi": 29.8, "children": 2, "smoker": "yes", "region": "southeast"}'

Invoke-WebRequest -Uri http://127.0.0.1:5000/predict -Method POST -ContentType "application/json" -Body $jsonData
```
