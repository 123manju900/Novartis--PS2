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

  
 






