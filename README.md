# Machine Failure Predictor

A Machine Learning web application that predicts whether a machine is likely to fail based on its operating conditions.

The application uses a **K-Nearest Neighbors (KNN)** classification model and **Streamlit** for the interactive web interface.

## Features

- Machine failure prediction using KNN
- Feature scaling using StandardScaler
- Train/test split with stratification
- Model accuracy evaluation
- Interactive Streamlit interface
- Real-time prediction based on machine parameters

## Input Features

| Feature | Description |
|---|---|
| Air Temperature [K] | Air temperature around the machine |
| Process Temperature [K] | Temperature of the machine process |
| Rotational Speed [rpm] | Machine rotational speed |
| Torque [Nm] | Torque applied to the machine |
| Tool Wear [min] | Accumulated tool usage time |

### Target Variable

`Machine failure`

- `0` — Machine is likely to operate normally
- `1` — Machine failure predicted

## Machine Learning Workflow

```text
Dataset
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
KNN Model Training
   ↓
Model Evaluation
   ↓
User Input
   ↓
Prediction
```

## Model

The project uses **K-Nearest Neighbors (KNN)** with `K = 5`.

```python
KNeighborsClassifier(
    n_neighbors=5
)
```

Since KNN is distance-based, `StandardScaler` is used to standardize the features before training and prediction.

## Project Structure

```text
machine-failure-predictor/
│
├── app.py
├── machine failure.csv
├── requirements.txt
└── README.md
```

## Technologies

- Python
- Pandas
- Scikit-learn
- Streamlit

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/machine-failure-predictor.git
```

Navigate to the project directory:

```bash
cd machine-failure-predictor
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

## Requirements

Create a `requirements.txt` file containing:

```text
streamlit
pandas
scikit-learn
```

## Model Evaluation

The dataset is divided into training and testing sets using an 80/20 split.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The application displays the model's test accuracy in the Streamlit interface.

## Prediction

Users can enter:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

The values are scaled using the same scaler used during model training and passed to the KNN model for prediction.

## Future Improvements

- Add confusion matrix and classification report
- Compare KNN with Random Forest and SVM
- Add prediction probability
- Add machine health indicators
- Improve the Streamlit interface
- Deploy the application using Streamlit Community Cloud

## Author

**Your Name**

Machine Learning / Data Science Project
