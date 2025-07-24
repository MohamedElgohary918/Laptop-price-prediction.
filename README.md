# 💻 Laptop Price Prediction

This project predicts the selling price of laptops based on detailed specifications using machine learning techniques.  
The model was trained on a dataset from **Kaggle**, and the best-performing model (Random Forest Regressor) was deployed using **Gradio** for real-time price predictions.

---

## 📊 Dataset
- **Source:** Kaggle (can also be collected via web scraping from e-commerce sites like Amazon).
- **Features:**
  - Brand and Model
  - RAM and Storage (SSD)
  - Processor (Generation and Core type)
  - Display size
  - Graphics (Integrated or Dedicated)
  - Operating System and Warranty
  - Selling Price and User Rating

---

## 🧹 Data Cleaning & Preprocessing
- ✅ Checked for null values and duplicates.
- ✅ Extracted numeric values from text columns (e.g., RAM, SSD).
- ✅ Standardized units (e.g., converted TB to GB).
- ✅ Dropped irrelevant or noisy columns.
- ✅ Normalized numerical columns for consistent scale.
- ✅ Applied Label Encoding to categorical columns.
- ✅ Analyzed correlations between features and price.

---

## ⚙️ Feature Engineering
- Extracted new features from existing columns.
- Created feature sets for better model performance.

---

## 🤖 Machine Learning Models
The following models were trained and evaluated:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor (SVR)

**Best Model:**  
`Random Forest Regressor` – achieved the highest R² score and lowest MAE.

---

## 🚀 Deployment
- The trained **Random Forest model** was deployed using **Gradio**, offering a simple, interactive interface for real-time laptop price prediction.


