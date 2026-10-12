import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ============================================
# 1. PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Machine Failure Predictor",
    layout="centered"
)


# ============================================
# 2. LOAD DATASET
# ============================================

df = pd.read_csv(r"machine failure.csv")


# ============================================
# 3. TITLE
# ============================================

st.title("Machine Failure Predictor")

st.write(
    "KNN-based machine failure prediction using "
    "machine operating conditions."
)


# ============================================
# 4. DATASET INFORMATION
# ============================================

with st.expander("View Dataset"):

    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================
# 5. SELECT FEATURES
# ============================================

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]

y = df["Machine failure"]


# ============================================
# 6. TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================
# 7. FEATURE SCALING
# ============================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================
# 8. CREATE KNN MODEL
# ============================================

model = KNeighborsClassifier(
    n_neighbors=5
)


# ============================================
# 9. TRAIN MODEL
# ============================================

model.fit(
    X_train_scaled,
    y_train
)


# ============================================
# 10. MODEL ACCURACY
# ============================================

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ============================================
# 11. DISPLAY MODEL PERFORMANCE
# ============================================

st.divider()

st.subheader("Model Performance")

st.metric(
    "Accuracy",
    f"{accuracy * 100:.2f}%"
)

st.write("K Value: 5")


# ============================================
# 12. MACHINE INPUT
# ============================================

st.divider()

st.subheader("Enter Machine Parameters")


air_temperature = st.number_input(
    "Air Temperature [K]",
    min_value=250.0,
    max_value=350.0,
    value=298.5,
    step=0.1
)


process_temperature = st.number_input(
    "Process Temperature [K]",
    min_value=250.0,
    max_value=400.0,
    value=308.8,
    step=0.1
)


rotational_speed = st.number_input(
    "Rotational Speed [rpm]",
    min_value=500,
    max_value=3000,
    value=1500,
    step=10
)


torque = st.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=100.0,
    value=45.0,
    step=0.1
)


tool_wear = st.number_input(
    "Tool Wear [min]",
    min_value=0,
    max_value=300,
    value=100,
    step=1
)


# ============================================
# 13. PREDICTION BUTTON
# ============================================

st.divider()

if st.button(
    "Predict Machine Failure",
    use_container_width=True
):

    # Create input DataFrame

    new_machine = pd.DataFrame({
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear]
    })


    # Scale input

    new_machine_scaled = scaler.transform(
        new_machine
    )


    # Make prediction

    prediction = model.predict(
        new_machine_scaled
    )


    # Display prediction

    st.subheader("Prediction Result")


    if prediction[0] == 1:

        st.error(
            "Machine Failure Predicted"
        )

    else:

        st.success(
            "Machine is Likely to Operate Normally"
        )
