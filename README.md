# 🩺 Breast Cancer Diagnosis Machine Learning Project

This repository contains a comprehensive machine learning project focused on classifying breast cancer tumors as either **benign** or **malignant**. It covers the entire data science pipeline — from extensive Exploratory Data Analysis (EDA) and preprocessing to preparing the data for robust model building.

---

## ✨ Project Overview

The primary goal of this project is to develop a predictive model that can accurately distinguish between benign and malignant breast cancer masses based on various cellular characteristics. This involves:

- Thorough data understanding and cleaning  
- Addressing class imbalance  
- Feature engineering and transformation  
- Dimensionality reduction to optimize the dataset for machine learning algorithms  

---

## 📊 Dataset Details

**Dataset**: Wisconsin Diagnostic Breast Cancer (WDBC) Dataset  
**Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  

**Description**:  
This dataset is derived from digitized images of a fine needle aspirate (FNA) of a breast mass. It describes characteristics of the cell nuclei present in the image.

**Key Attributes**:

- **ID Number**: Unique identifier for each sample (removed during preprocessing)  
- **Diagnosis (Target Variable)**: M (Malignant) or B (Benign)  
- **30 Real-valued Features**: Computed for each cell nucleus and derived from 10 core features:

  Each of these has three variations:
  - `_mean`: The mean value
  - `_se`: The standard error
  - `_worst`: The worst/largest value (mean of the three largest values)

---

## 🔬 Exploratory Data Analysis (EDA) & Preprocessing

### ✅ Step 1: Data Loading and Initial Inspection
- Loaded `wdbc.data` into a Pandas DataFrame  
- Assigned column names based on `wdbc.names` metadata  
- Used `df.head()`, `df.info()`, `df.describe()` for initial inspection  

**Observations**:
- 569 entries, 32 columns, no missing values  
- ID column is non-predictive  
- Feature scaling required due to varying ranges  

---

### ✅ Step 2: Data Cleaning and Encoding
- Dropped the ID column  
- Encoded 'Diagnosis' column:  
  - 0 for Benign  
  - 1 for Malignant  
- Removed original categorical column  
<img width="540" height="393" alt="image" src="https://github.com/user-attachments/assets/4e9d2145-bf3c-4f63-a0b4-b2c559206f95" />

**Observations**:
- Streamlined dataset  
- Target variable made ML-compatible  

---

### ✅ Step 3: Class Imbalance Handling (SMOTE)
- Split data: 80% train / 20% test (stratified)  
- Applied **SMOTE** only on training data  
<img width="663" height="470" alt="image" src="https://github.com/user-attachments/assets/31e4457e-e1ec-4f9d-85ab-c26684d60acf" />

**Observations**:
- Initial imbalance: 285 Benign / 170 Malignant  
- Post-SMOTE: 285 each — **perfectly balanced**  
- Test set remained untouched for unbiased evaluation  

---

### ✅ Step 4: Feature Scaling
- Applied `StandardScaler` on training & test data  
- Fitted only on training data to avoid data leakage  

**Observations**:
- All features now have mean ≈ 0, std ≈ 1  
- Essential for algorithms like SVM, KNN, Logistic Regression, etc.  

---

### ✅ Step 5: Feature Distribution Visualization
- Created histograms and KDE plots for selected features  
<img width="1589" height="1389" alt="image" src="https://github.com/user-attachments/assets/ecaf7c0b-98b9-4338-b780-a67b7ed9038b" />

**Observations**:
- _se features still had high skewness and kurtosis  
- Some features were fairly symmetrical  

---

### ✅ Step 6: Outlier Investigation (Box Plots)
- Visualized outliers across features using boxplots  
<img width="1379" height="1489" alt="image" src="https://github.com/user-attachments/assets/cc20070d-c240-403e-b14d-7f44d14ebaab" />

**Observations**:
- Several outliers, especially in skewed features  
- Contributed to non-normality of distributions  

---

### ✅ Step 7: Correlation Analysis
- Computed correlation matrix  
- Visualized via heatmap  
- Identified top 10 most correlated features  
<img width="1842" height="1790" alt="image" src="https://github.com/user-attachments/assets/03f85d13-5b0a-462c-9063-84022213371d" />

**Observations**:
- Strong predictors: `concave points_worst`, `radius_mean`, `area_mean`, etc.  
- High multicollinearity among several features  

---

### ✅ Step 8: Feature Transformation (Logarithmic Scaling)
- Applied `np.log1p` to skewed features:  
  `area_se`, `concavity_se`, `fractal_dimension_se`, `perimeter_se`, `radius_se`, `smoothness_se`, `symmetry_se`  
<img width="1188" height="490" alt="image" src="https://github.com/user-attachments/assets/14dac8d6-a2cd-4cb3-90de-952ba70391bc" />
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/00346af4-7b09-4663-8298-eeebf7604954" />
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/3ac5dd83-6f1c-4dc7-93a9-5a0c58ad0b5c" />
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/02cb5d76-99b4-4a11-ac38-19711c1702cb" />
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/62b290a4-1563-44cb-93c6-d5f50e553d9d" />
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/4ee97cdd-3e9f-4f60-9906-a27538cf93f7" />
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/b78c55de-024a-4a2b-900a-aaeb4be4c5da" />

**Observations**:
- Significantly reduced skewness  
- Some over-transformed (now negatively skewed)  
- Runtime warnings due to log1p on values < 0  

> *(Optional)*: You can insert your own transformation plots like:  
> `![concavity_se Transformation Plot](YOUR_CONCAVITY_SE_TRANSFORMATION_PLOT_URL_HERE)`

---

### ✅ Step 9: Dimensionality Reduction (PCA)
- Applied **PCA** on scaled data  
- Used cumulative explained variance to select components  
<img width="846" height="547" alt="image" src="https://github.com/user-attachments/assets/db72dec7-4d7f-43e3-ae27-3c816d9dbddd" />

**Observations**:
- Reduced from 30 to 10 components  
- Explained ≥ 95% variance  
- Mitigated multicollinearity and improved model efficiency  
- Final shapes:  
  - `X_train_pca`: (570, 10)  
  - `X_test_pca`: (114, 10)

---

## 🚀 Future Work / Next Steps

1. **Model Selection**:  
   Trying algorithms like:
   - Logistic Regression  
   - SVM  
   - Decision Tree  
   - Random Forest  
   - XGBoost / LightGBM  

2. **Model Training**:  
   Train on `X_train_pca` and `y_train_resampled`

3. **Model Evaluation**:  
   Use:
   - Accuracy  
   - Precision  
   - Recall  
   - F1-Score  
   - Confusion Matrix  
   - ROC AUC  

4. **Hyperparameter Tuning**:  
   Use Grid Search or Randomized Search with Cross-Validation  

5. **Model Interpretation**:  
   Feature importance, SHAP plots, etc.

6. **Deployment (Optional)**:  
   Explore Streamlit, Flask, or FastAPI for real-time predictions  

---
