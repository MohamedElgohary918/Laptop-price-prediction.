import gradio as gr
import joblib
import pandas as pd

# === Load Saved Model, Scaler, and Encoders ===
model = joblib.load("notebooks/models/laptop_price_model.pkl")
scaler = joblib.load("notebooks/models/normalizer.pkl")
label_encoders = joblib.load("notebooks/models/label_encoders.pkl")
numerical_cols = joblib.load("notebooks/models/numerical_cols.pkl")
price_scaler = joblib.load("notebooks/models/price_scaler.pkl")


# === Define Prediction Function ===
def predict_price(Company, TypeName, Ram, Weight, IPS, ppi, Cpu_Brand, SSD, HDD, Gpu_Brand, OS):
    print("Received inputs")

    # Step 1: Create input DataFrame
    input_data = pd.DataFrame([[
        Company, TypeName, Ram, Weight, IPS, ppi, Cpu_Brand, SSD, HDD, Gpu_Brand, OS
    ]], columns=[
        'Company', 'TypeName', 'Ram', 'Weight', 'IPS', 'ppi',
        'Cpu Brand', 'SSD', 'HDD', 'Gpu Brand', 'OS'
    ])

    
    # Step 2: Convert numeric types
    input_data['Ram'] = int(Ram)
    input_data['Weight'] = float(Weight)
    input_data['IPS'] = int(IPS)
    input_data['ppi'] = float(ppi)
    input_data['SSD'] = int(SSD)
    input_data['HDD'] = int(HDD)

    # Step 3: Label encode categorical features
    try:
        for col in input_data.select_dtypes(include=['object']).columns:
            le = label_encoders.get(col)
            if le:
                if input_data[col][0] not in le.classes_:
                    print(f"Unknown category in column {col}: {input_data[col][0]}")
                    return f"Invalid input for '{col}': '{input_data[col][0]}'"
                input_data[col] = le.transform(input_data[col])
    except Exception as e:
        print(f"Encoding error: {e}")
        return "Error during label encoding."

    # Step 4: Scale numeric features using saved column order
    try:
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
    except Exception as e:
        print(f"Scaling error: {e}")
        return f"Error during scaling: {e}"

# Predict
    try:
        prediction_scaled = model.predict(input_data)[0]
        prediction = price_scaler.inverse_transform([[prediction_scaled]])[0][0]  
        print(" Prediction done")
        return f" Predicted Price: €{round(prediction, 2)}"
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error during prediction."


# === Gradio Interface ===
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Dropdown(
            ["Apple", "HP", "Acer", "Asus", "Dell", "Lenovo", "Chuwi", "MSI", "Microsoft",
             "Toshiba", "Huawei", "Xiaomi", "Vero", "Razer", "Mediacom", "Samsung", "Google",
             "Fujitsu", "LG"],
            label="Company"
        ),
        gr.Dropdown(
            ["Ultrabook", "Notebook", "Netbook", "Gaming", "2 in 1 Convertible", "Workstation"],
            label="TypeName"
        ),
        gr.Number(label="Ram (GB)"),
        gr.Number(label="Weight (kg)"),
        gr.Radio([0, 1], label="IPS Panel (1=Yes, 0=No)"),
        gr.Number(label="PPI (Pixels Per Inch)"),
        gr.Dropdown(
            ["Intel Core i5", "Intel Core i7", "AMD Processor", "Intel Core i3", "Other Intel Processor"],
            label="CPU Brand"
        ),
        gr.Number(label="SSD (GB)"),
        gr.Number(label="HDD (GB)"),
        gr.Dropdown(["Intel", "AMD", "Nvidia"], label="GPU Brand"),
        gr.Dropdown(["Mac", "Others/No OS/Linux", "Windows"], label="Operating System (OS)")
    ],
    outputs="text",
    title="💻 Laptop Price Predictor",
    description="Enter laptop specifications to predict the price (€) using a machine learning model."
)

# === Launch the app ===
iface.launch()
