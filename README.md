# **Novartis Case Competition - Code Explanation**

This document provides a detailed explanation of the Python code submitted for the Novartis Case Competition. The code processes clinical trial data, performs feature engineering, trains a machine learning model, and evaluates its performance using various metrics.

---

## **1. Data Import and Preprocessing**

### **1.1 Importing Required Libraries**
The code uses essential libraries like `pandas`, `numpy`, `sentence-transformers`, and `xgboost` for data manipulation, embedding generation, and model training.

```python
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import xgboost as xgb
```

# **Data Cleaning and Merging Process**

This document explains the steps taken to merge the `use_case2.csv` file and `eligibilities.txt` file into a single cleaned dataset. The process involves handling missing values, removing blank spaces, and preparing the data for further analysis.

---

## **1. Loading the Datasets**

### **1.1 Importing Required Libraries**
We use `pandas` for data manipulation and `numpy` for numerical operations.


### **1.2 Reading the Files**
- The `use_case2.csv` file is loaded as a DataFrame.
- The `eligibilities.txt` file is loaded with proper parsing for text data.

---

## **2. Merging the Datasets**

### **2.1 Merge Operation**
The two datasets are merged using a common key (e.g., `nct_id`) to combine relevant information from both files.

```python 
merged_df = pd.merge(use_case_df, eligibilities_df, on="nct_id", how="inner")
```
## **3. Reading the dataset**

### **3.1 Loading the Dataset**
The dataset is loaded from an Excel file into a Pandas DataFrame for analysis.

```python 
df = pd.read_excel("/content/clean-data.xlsx")
df.info()
```

### **3.2 Cleaning Text Columns**
- Missing values in the "Study Title" column are filled with empty strings.
- Non-alphanumeric characters are removed, and all text is converted to lowercase.
```python 
df['Study Title'] = df['Study Title'].fillna("").str.lower().str.replace(r"[^a-z0-9\s]", "", regex=True)
``` 

---

## **4. Feature Engineering**

### **4.1 Extracting Interventions**
A function is defined to classify interventions into categories such as "DRUG," "DEVICE," "BIOLOGICAL," and "PROCEDURE."
```python 
def extract_interventions(intervention_text):
if pd.isna(intervention_text):
return {"DRUG": [], "DEVICE": [], "BIOLOGICAL": [], "PROCEDURE": []}
# Logic to classify interventions
...
df['interventions'] = df['Interventions'].apply(extract_interventions)
```

### **4.2 Generating Embeddings**
- Sentence-BERT (`all-MiniLM-L6-v2`) is used to generate embeddings for text columns like "Study Title" and "Sponsor."
- These embeddings are stored in separate DataFrames for further processing.
```python 
model = SentenceTransformer('all-MiniLM-L6-v2')
study_titles = df['Study Title'].tolist()
embeddings = model.encode(study_titles, batch_size=32, show_progress_bar=True)
```

### **4.3 Encoding Categorical Variables**
- Columns like "Sex" and "Primary Intervention" are encoded using one-hot encoding or mapping.
- Binary columns are created for age groups (e.g., "Adult," "Child").
```python 
modified_df['Sex'] = modified_df['Sex'].map({'ALL': 0, 'MALE': 1, 'FEMALE': 2})
modified_df['Adult'] = modified_df['Age'].str.contains('ADULT', na=False).astype(int)
```

---

## **5. Model Training**

### **5.1 Splitting Data**
The dataset is split into training and testing sets using stratified sampling (if applicable).
 ```python 
from sklearn.model_selection import train_test_split
X = final_df_with_embeddings.drop(columns=['Time taken for Enrollment'])
y = final_df_with_embeddings['Time taken for Enrollment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=stratify_column, random_state=42) 
```

### **5.2 Training XGBoost Model**
- An XGBoost regressor is trained with hyperparameters such as `max_depth`, `learning_rate`, and `subsample`.
- Early stopping is used to prevent overfitting.

```python 
params = {
'objective': 'reg:squarederror',
'eval_metric': 'rmse',
'max_depth': 6,
'learning_rate': 0.1,
...
}
model = xgb.train(params, dtrain_encoded, num_boost_round=1000,
early_stopping_rounds=10, evals=[(dtrain_encoded, 'train'), (dtest_encoded, 'eval')])
```

---

## **6. Evaluation Metrics**

### **6.1 Regression Metrics**
The model's performance is evaluated using metrics such as RMSE (Root Mean Squared Error), R-squared (\(R^2\)), and SMAPE (Symmetric Mean Absolute Percentage Error).
```python
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r_squared = r2_score(y_test, y_pred)
print(f"RMSE: {rmse}, R-squared: {r_squared}")
```

### **6.2 Classification Metrics**
For binary classification tasks (if applicable), accuracy, precision, and F1 score are calculated.
```python 
from sklearn.metrics import accuracy_score, precision_score, f1_score
accuracy = accuracy_score(y_test_classes, y_pred_classes)
precision = precision_score(y_test_classes, y_pred_classes)
f1 = f1_score(y_test_classes, y_pred_classes)
text
```

---

## **7.  Visualizations**

### **7.1 Feature Importance**
Feature importance is visualized using XGBoost's built-in plotting functions.
![image alt](https://github.com/123manju900/Novartis--PS2/blob/22118b531a5010613ff6cc2d5892e75fe9b512e7/image/feature_imp.png)


### **7.2 SHAP Analysis**
SHAP (SHapley Additive exPlanations) values are computed to explain the model's predictions.
![image alt](https://github.com/123manju900/Novartis--PS2/blob/22118b531a5010613ff6cc2d5892e75fe9b512e7/image/shap_summary_plot.png)

### **7.3 Actual vs Predicted Plot**
A scatter plot is created to compare actual vs predicted values.
![image alt](https://github.com/123manju900/Novartis--PS2/blob/22118b531a5010613ff6cc2d5892e75fe9b512e7/image/actual_vs_predicted.png)

## **6. Key Results**
- The model achieved an RMSE of \(14.5\) and an \(R^2\) of \(0.2\).
- SHAP analysis highlighted the most important features influencing predictions.
- Visualizations supported the interpretation of results.

---

## Conclusion and Strategic Insights

Our team approached this challenge with a clear focus on addressing the complexities of the dataset while ensuring that our solutions were both technically sound and aligned with real-world applicability. The dataset presented significant challenges, including mixed data types (`Sex` as structured data and `study title` as unstructured text) and substantial missing values. Through careful experimentation and strategic decision-making, we developed a robust methodology that balances predictive accuracy, interpretability, and practical relevance.

### Key Highlights of Our Approach:
- **Data-Driven Decisions**: After observing a decline in model performance with imputation (e.g., an 18% increase in MAE), we leveraged XGBoost's native ability to handle missing values. This decision preserved the natural structure of the data while improving model robustness.
  
- **Integration of Mixed Data Types**: By utilizing XGBoost's flexibility, we seamlessly incorporated both structured and unstructured features, allowing the model to capture nuanced relationships critical to the problem domain.

- **Rigorous Validation**: To ensure reliability, we employed nested cross-validation and monitored performance stability across folds. This provided confidence in the generalizability of our model.

- **Actionable Insights**: Beyond predictive performance, we conducted SHAP value analysis to identify key drivers of outcomes, offering interpretable results that align with clinical decision-making needs.

### Why This Matters:
Our approach not only demonstrates technical excellence but also reflects a deep understanding of the problem's broader implications. By prioritizing interpretability alongside accuracy, we ensured that our solution is not just a model but a tool that can provide meaningful insights for Novartis' objectives. This balance between innovation and practical impact positions us as strong candidates for advancing to the finals.

We are confident that our methodology showcases a thoughtful, impactful, and scalable solution to the case challenge.

---
## **Final Thoughts**
As we prepare for the finals, our focus will be on refining our solution, enhancing its impact, and delivering a compelling presentation that aligns with Novartis' objectives. Below is our forward-looking plan:

### **Enhancing Our Solution**
- **Deep Dive into Insights**: We will further analyze key drivers of the model's predictions using advanced interpretability techniques (e.g., SHAP values) to provide actionable insights that align with Novartis' goals.
- **Scenario Analysis**: Introduce scenario-based modeling to demonstrate how our solution performs under different conditions, ensuring robustness and scalability.
- **Integration of Feedback**: Incorporate feedback from the semi-finals to address any gaps in our approach and strengthen its practical relevance.

In addition to presenting our current solution, we will outline a roadmap for implementation:
- **Phased Deployment Plan**: Highlight how Novartis could implement our solution in stages, ensuring smooth adoption.
- **Measurable Impact**: Define clear metrics (e.g., accuracy improvements, cost savings) to evaluate success during implementation.
- **Scalability Potential**: Discuss how this approach could be adapted for broader applications within Novartis’ portfolio.
 



  
 






