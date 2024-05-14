# Credit Score Evaluation Using Stacking Classifier
### Kagle DataSet
https://www.kaggle.com/datasets/parisrohan/credit-score-classification

## Final Model Performance Scores

<table border="1">
  <tr>
    <th>Credit Score</th>
    <th>precision</th>
    <th>recall</th>
    <th>f1-score</th>
    <th>support</th>
  </tr>
  <tr>
    <td>Poor</td>
    <td>0.88</td>
    <td>0.88</td>
    <td>0.88</td>
    <td>15990</td>
  </tr>
  <tr>
    <td>Standard</td>
    <td>0.80</td>
    <td>0.82</td>
    <td>0.81</td>
    <td>15658</td>
  </tr>
  <tr>
    <td>Good</td>
    <td>0.92</td>
    <td>0.90</td>
    <td>0.91</td>
    <td>16209</td>
  </tr>
</table>

### Overall Performance

<table border="1">
  <tr>
    <td>accuracy:</td>
    <td>0.87</td>
  </tr>
  <tr>
    <td>macro avg:</td>
    <td>0.87</td>
  </tr>
  <tr>
    <td>weighted avg:</td>
    <td>0.87</td>
  </tr>
  <tr>
    <td>support:</td>
    <td>47857</td>
  </tr>
</table>


```python
# Packages for EDA 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
import numpy as np 
import string
import os

# Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from datasist.structdata import detect_outliers
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import category_encoders as ce
import re 

# Outlier Transformer
from sklearn.base import BaseEstimator, TransformerMixin

# Visulalization
%matplotlib inline
matplotlib.rc(("xtick", "ytick", "text"), c="k")
matplotlib.rc("figure", dpi=80)

# Modeling and evaluation 
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
    StackingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier
)
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
import joblib

# Packages options 
sns.set(rc={'figure.figsize': [14, 7]}, font_scale=1.2) # Standard figure size for all 
np.seterr(divide='ignore', invalid='ignore', over='ignore') ;

import warnings 
warnings.filterwarnings("ignore")
```

    /Users/duleepp/anaconda3/lib/python3.11/site-packages/sklearn/experimental/enable_hist_gradient_boosting.py:16: UserWarning: Since version 1.0, it is not needed to import enable_hist_gradient_boosting anymore. HistGradientBoostingClassifier and HistGradientBoostingRegressor are now stable and can be normally imported from sklearn.ensemble.
      warnings.warn(



```python
train_df = pd.read_csv("./train.csv", low_memory=False)
train_df["is_train"] = True
test_df = pd.read_csv("./test.csv", low_memory=False)
test_df["is_train"] = False

df = pd.concat([train_df, test_df])
```

# Data Exploration


```python
df.columns
```




    Index(['ID', 'Customer_ID', 'Month', 'Name', 'Age', 'SSN', 'Occupation',
           'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
           'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Type_of_Loan',
           'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
           'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
           'Credit_Utilization_Ratio', 'Credit_History_Age',
           'Payment_of_Min_Amount', 'Total_EMI_per_month',
           'Amount_invested_monthly', 'Payment_Behaviour', 'Monthly_Balance',
           'Credit_Score', 'is_train'],
          dtype='object')




```python
# Dropping irrelevant columns
df.drop(["Name", "SSN", "ID"], axis=1, inplace=True, errors="ignore")
```


```python
# Change dtype for specific columns
columns_to_convert = ["Month", "Occupation", "Type_of_Loan", "Credit_History_Age", "Payment_Behaviour"]
df[columns_to_convert] = df[columns_to_convert].astype("category")
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 150000 entries, 0 to 49999
    Data columns (total 26 columns):
     #   Column                    Non-Null Count   Dtype   
    ---  ------                    --------------   -----   
     0   Customer_ID               150000 non-null  object  
     1   Month                     150000 non-null  category
     2   Age                       150000 non-null  object  
     3   Occupation                150000 non-null  category
     4   Annual_Income             150000 non-null  object  
     5   Monthly_Inhand_Salary     127500 non-null  float64 
     6   Num_Bank_Accounts         150000 non-null  int64   
     7   Num_Credit_Card           150000 non-null  int64   
     8   Interest_Rate             150000 non-null  int64   
     9   Num_of_Loan               150000 non-null  object  
     10  Type_of_Loan              132888 non-null  category
     11  Delay_from_due_date       150000 non-null  int64   
     12  Num_of_Delayed_Payment    139500 non-null  object  
     13  Changed_Credit_Limit      150000 non-null  object  
     14  Num_Credit_Inquiries      147000 non-null  float64 
     15  Credit_Mix                150000 non-null  object  
     16  Outstanding_Debt          150000 non-null  object  
     17  Credit_Utilization_Ratio  150000 non-null  float64 
     18  Credit_History_Age        136500 non-null  category
     19  Payment_of_Min_Amount     150000 non-null  object  
     20  Total_EMI_per_month       150000 non-null  float64 
     21  Amount_invested_monthly   143250 non-null  object  
     22  Payment_Behaviour         150000 non-null  category
     23  Monthly_Balance           148238 non-null  object  
     24  Credit_Score              100000 non-null  object  
     25  is_train                  150000 non-null  bool    
    dtypes: bool(1), category(5), float64(4), int64(4), object(12)
    memory usage: 25.4+ MB



```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Monthly_Inhand_Salary</th>
      <th>Num_Bank_Accounts</th>
      <th>Num_Credit_Card</th>
      <th>Interest_Rate</th>
      <th>Delay_from_due_date</th>
      <th>Num_Credit_Inquiries</th>
      <th>Credit_Utilization_Ratio</th>
      <th>Total_EMI_per_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>127500.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
      <td>147000.000000</td>
      <td>150000.000000</td>
      <td>150000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4190.115139</td>
      <td>17.006940</td>
      <td>22.623447</td>
      <td>71.234907</td>
      <td>21.063400</td>
      <td>28.529014</td>
      <td>32.283309</td>
      <td>1432.513579</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3180.489657</td>
      <td>117.069476</td>
      <td>129.143006</td>
      <td>461.537193</td>
      <td>14.860154</td>
      <td>194.456058</td>
      <td>5.113315</td>
      <td>8403.759977</td>
    </tr>
    <tr>
      <th>min</th>
      <td>303.645417</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>-5.000000</td>
      <td>0.000000</td>
      <td>20.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1625.265833</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
      <td>28.054731</td>
      <td>30.947775</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3091.000000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>13.000000</td>
      <td>18.000000</td>
      <td>6.000000</td>
      <td>32.297058</td>
      <td>71.280006</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5948.454596</td>
      <td>7.000000</td>
      <td>7.000000</td>
      <td>20.000000</td>
      <td>28.000000</td>
      <td>9.000000</td>
      <td>36.487954</td>
      <td>166.279555</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15204.633333</td>
      <td>1798.000000</td>
      <td>1499.000000</td>
      <td>5799.000000</td>
      <td>67.000000</td>
      <td>2597.000000</td>
      <td>50.000000</td>
      <td>82398.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.duplicated().sum()
```




    0




```python
df.isna().sum()
```




    Customer_ID                     0
    Month                           0
    Age                             0
    Occupation                      0
    Annual_Income                   0
    Monthly_Inhand_Salary       22500
    Num_Bank_Accounts               0
    Num_Credit_Card                 0
    Interest_Rate                   0
    Num_of_Loan                     0
    Type_of_Loan                17112
    Delay_from_due_date             0
    Num_of_Delayed_Payment      10500
    Changed_Credit_Limit            0
    Num_Credit_Inquiries         3000
    Credit_Mix                      0
    Outstanding_Debt                0
    Credit_Utilization_Ratio        0
    Credit_History_Age          13500
    Payment_of_Min_Amount           0
    Total_EMI_per_month             0
    Amount_invested_monthly      6750
    Payment_Behaviour               0
    Monthly_Balance              1762
    Credit_Score                50000
    is_train                        0
    dtype: int64




```python
df.describe(exclude=np.number).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Customer_ID</th>
      <td>150000</td>
      <td>12500</td>
      <td>CUS_0xd40</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Month</th>
      <td>150000</td>
      <td>12</td>
      <td>April</td>
      <td>12500</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>150000</td>
      <td>2524</td>
      <td>39</td>
      <td>4198</td>
    </tr>
    <tr>
      <th>Occupation</th>
      <td>150000</td>
      <td>16</td>
      <td>_______</td>
      <td>10500</td>
    </tr>
    <tr>
      <th>Annual_Income</th>
      <td>150000</td>
      <td>21192</td>
      <td>36585.12</td>
      <td>24</td>
    </tr>
    <tr>
      <th>Num_of_Loan</th>
      <td>150000</td>
      <td>623</td>
      <td>3</td>
      <td>21500</td>
    </tr>
    <tr>
      <th>Type_of_Loan</th>
      <td>132888</td>
      <td>6260</td>
      <td>Not Specified</td>
      <td>2112</td>
    </tr>
    <tr>
      <th>Num_of_Delayed_Payment</th>
      <td>139500</td>
      <td>1058</td>
      <td>19</td>
      <td>7949</td>
    </tr>
    <tr>
      <th>Changed_Credit_Limit</th>
      <td>150000</td>
      <td>4605</td>
      <td>_</td>
      <td>3150</td>
    </tr>
    <tr>
      <th>Credit_Mix</th>
      <td>150000</td>
      <td>4</td>
      <td>Standard</td>
      <td>54858</td>
    </tr>
    <tr>
      <th>Outstanding_Debt</th>
      <td>150000</td>
      <td>13622</td>
      <td>1360.45</td>
      <td>36</td>
    </tr>
    <tr>
      <th>Credit_History_Age</th>
      <td>136500</td>
      <td>408</td>
      <td>17 Years and 11 Months</td>
      <td>628</td>
    </tr>
    <tr>
      <th>Payment_of_Min_Amount</th>
      <td>150000</td>
      <td>3</td>
      <td>Yes</td>
      <td>78484</td>
    </tr>
    <tr>
      <th>Amount_invested_monthly</th>
      <td>143250</td>
      <td>136497</td>
      <td>__10000__</td>
      <td>6480</td>
    </tr>
    <tr>
      <th>Payment_Behaviour</th>
      <td>150000</td>
      <td>7</td>
      <td>Low_spent_Small_value_payments</td>
      <td>38207</td>
    </tr>
    <tr>
      <th>Monthly_Balance</th>
      <td>148238</td>
      <td>148224</td>
      <td>__-333333333333333333333333333__</td>
      <td>15</td>
    </tr>
    <tr>
      <th>Credit_Score</th>
      <td>100000</td>
      <td>3</td>
      <td>Standard</td>
      <td>53174</td>
    </tr>
    <tr>
      <th>is_train</th>
      <td>150000</td>
      <td>2</td>
      <td>True</td>
      <td>100000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get Unique Values
def get_unique_values(df):
    cat_cols = df.select_dtypes("object").columns

    data_info = np.zeros((len(cat_cols), 5), dtype="object")
    for i, col in enumerate(cat_cols):
        if len(df[col].unique()) > 5000:
            continue
        else:
            unique_values, counts = np.unique(
                np.array(df[col], dtype=str), return_counts=True)
            num_of_uv = len(unique_values)
            unique_val_percent = np.round(counts / counts.sum(), 2)
            data_info[i, :] = [col, unique_values.tolist(
            ), counts.tolist(), num_of_uv, unique_val_percent]
    return pd.DataFrame(data_info, columns=["column", "unique", "counts", "len_unique_values", "%_unique_values"])
```


```python
unique_values_df = get_unique_values(df)
unique_values_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column</th>
      <th>unique</th>
      <th>counts</th>
      <th>len_unique_values</th>
      <th>%_unique_values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Age</td>
      <td>[-500, 100, 1004, 1006, 1007, 1010, 1018_, 102...</td>
      <td>[1350, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...</td>
      <td>2524</td>
      <td>[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Num_of_Loan</td>
      <td>[-100, 0, 0_, 1, 100, 1001, 1002, 1006, 1008, ...</td>
      <td>[5850, 15543, 833, 15112, 1, 1, 1, 1, 1, 1, 1,...</td>
      <td>623</td>
      <td>[0.04, 0.1, 0.01, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Num_of_Delayed_Payment</td>
      <td>[-1, -1_, -2, -2_, -3, -3_, 0, 0_, 1, 10, 100,...</td>
      <td>[431, 12, 326, 17, 140, 5, 2352, 68, 2400, 767...</td>
      <td>1059</td>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.02...</td>
    </tr>
  </tbody>
</table>
</div>



# Data Processing


```python
class DataProcessor:
    def __init__(self, groupby, data_frame):
        self.groupby = groupby
        self.df = data_frame

    def get_month(self, x):
     if not pd.isnull(x):
         year_month = re.findall(r"\d+", x)
         months = (int(year_month[0])*12) + np.int64(year_month[-1])
         return months
     else:
         x

    @staticmethod
    def get_numbers(text):
        digits = re.findall(r'\d+', str(text))
        digits = ','.join(digits)
        return digits

    @staticmethod
    def replace_special_character(text):
        if "NM" in str(text):
            return "No"
        if "payments" in str(text) or "_" not in str(text):
            return text
        clean_text = str(text).replace("_", "")
        return np.nan if clean_text == "nan" else clean_text

    @staticmethod
    def preprocess_text(texts:str) -> tuple[dict, list[list[str]]]:
        dictionary = {}
        tokens = [str(text).lower().replace("and", "").split(",") for text in texts]
        tokens = [[token.strip() for token in token_list if token not in string.punctuation] for token_list in tokens]
        for token_list in tokens:
            for token in token_list:
                if token not in dictionary:
                    size = len(dictionary)
                    dictionary[token] = size
        return (dictionary, ["|".join(words) for words in tokens])


    @staticmethod
    def fill_na(df: pd.DataFrame, groupby=None):
        cat_features = df.select_dtypes(exclude="number").columns.drop(
            ["Type_of_Loan", "is_train"])
        num_features = df.select_dtypes(include="number").columns
        df["Type_of_Loan"].fillna("not specified", inplace=True)
        if "Credit_Score" in df.columns:
            cat_features = cat_features.drop("Credit_Score")
        
        # Replacing Categorial Columns with Mode
        def fill_na_cat(df):
            df[cat_features] = df.groupby(groupby)[cat_features].transform(
                lambda x: x.fillna(x.mode()[0]))
            return df
        
        # Replacing Numerical Columns with Median
        def fill_na_num(df):
            df[num_features] = df.groupby(groupby)[num_features].transform(
                lambda x: x.mask(x < 0, np.nan).fillna(x.median()))
            return df

        df = fill_na_cat(df)
        df = fill_na_num(df)
        return df
    def preprocess(self):
        # Age
        self.df['Age'] = self.df.Age.apply(DataProcessor.get_numbers)
        # Handle Special Characters
        self.df = self.df.applymap(DataProcessor.replace_special_character)
        self.df = self.df.apply(lambda x: pd.to_numeric(x, errors="ignore"))
        # Credit Mix
        self.df["Credit_Mix"] = self.df.groupby(self.groupby)["Credit_Mix"].transform(lambda x: x.replace("", x.mode()[0]))
        # Payment Behaviour
        self.df["Payment_Behaviour"] = self.df.groupby(self.groupby)["Payment_Behaviour"].transform(
            lambda x: x.replace("!@9#%8", x.mode()[0])
        )
        self.df["Payment_Behaviour"] = self.df["Payment_Behaviour"].transform(
            lambda x: x.replace("!@9#%8", x.mode()[0])
        )
        # Type of Loan
        self.df["Type_of_Loan"] = self.df[["Type_of_Loan"]].apply(lambda x:  DataProcessor.preprocess_text(x.values)[-1])
        self.df["Type_of_Loan"] = self.df["Type_of_Loan"].str.replace(" ", "_").str.replace("|", " ").replace("nan", np.nan)
        # Credit History Age
        self.df["Credit_History_Age"] = self.df["Credit_History_Age"].apply(lambda x: self.get_month(x))
        # Monthly Balance
        self.df["Monthly_Balance"] = pd.to_numeric(self.df.Monthly_Balance, errors="coerce")
        # Replacing account balances less than zero with zero
        self.df.loc[self.df["Num_Bank_Accounts"] < 0, "Num_Bank_Accounts"] = 0
        # Replace "nan" values in the 'Type_of_Loan' column with NaN for consistency
        self.df.loc[self.df["Type_of_Loan"] == "nan", "Type_of_Loan"] = np.nan
        # Replace "nan" values in the 'Occupation' column with NaN for consistency
        self.df.loc[self.df["Occupation"] == "", "Occupation"] = np.nan
        self.df.loc[self.df["Occupation"] == "_______", "Occupation"] = np.nan
        # Replace "nan" values in the 'Credit_Mix' column with NaN for consistency
        self.df.loc[self.df["Credit_Mix"] == "", "Credit_Mix"] = np.nan

        # Negetive Numbers
        self.df['Num_of_Delayed_Payment'] = pd.to_numeric(self.df['Num_of_Delayed_Payment'], errors='coerce')

        self.df.loc[self.df['Num_of_Delayed_Payment'] < 0, 'Num_of_Delayed_Payment'] = np.nan
        self.df.loc[self.df['Delay_from_due_date'] < 0, 'Delay_from_due_date'] = np.nan
        
        # Filling missing values
        self.df = DataProcessor.fill_na(self.df, "Customer_ID")

        return self.df
```


```python
preprocesor = DataProcessor("Customer_ID", df)
new_df = preprocesor.preprocess()
```


```python
new_df.isna().sum()
```




    Customer_ID                     0
    Month                           0
    Age                             0
    Occupation                      0
    Annual_Income                   0
    Monthly_Inhand_Salary           0
    Num_Bank_Accounts               0
    Num_Credit_Card                 0
    Interest_Rate                   0
    Num_of_Loan                     0
    Type_of_Loan                    0
    Delay_from_due_date             0
    Num_of_Delayed_Payment          0
    Changed_Credit_Limit            0
    Num_Credit_Inquiries            0
    Credit_Mix                      0
    Outstanding_Debt                0
    Credit_Utilization_Ratio        0
    Credit_History_Age              0
    Payment_of_Min_Amount           0
    Total_EMI_per_month             0
    Amount_invested_monthly         0
    Payment_Behaviour               0
    Monthly_Balance                 0
    Credit_Score                50000
    is_train                        0
    dtype: int64




```python
new_df[new_df.isna().any(axis=1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer_ID</th>
      <th>Month</th>
      <th>Age</th>
      <th>Occupation</th>
      <th>Annual_Income</th>
      <th>Monthly_Inhand_Salary</th>
      <th>Num_Bank_Accounts</th>
      <th>Num_Credit_Card</th>
      <th>Interest_Rate</th>
      <th>Num_of_Loan</th>
      <th>...</th>
      <th>Outstanding_Debt</th>
      <th>Credit_Utilization_Ratio</th>
      <th>Credit_History_Age</th>
      <th>Payment_of_Min_Amount</th>
      <th>Total_EMI_per_month</th>
      <th>Amount_invested_monthly</th>
      <th>Payment_Behaviour</th>
      <th>Monthly_Balance</th>
      <th>Credit_Score</th>
      <th>is_train</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CUS0xd40</td>
      <td>September</td>
      <td>23</td>
      <td>Scientist</td>
      <td>19114.12</td>
      <td>1824.843333</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>4.0</td>
      <td>...</td>
      <td>809.98</td>
      <td>35.030402</td>
      <td>273.0</td>
      <td>No</td>
      <td>49.574949</td>
      <td>236.642682</td>
      <td>Low_spent_Small_value_payments</td>
      <td>186.266702</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CUS0xd40</td>
      <td>October</td>
      <td>24</td>
      <td>Scientist</td>
      <td>19114.12</td>
      <td>1824.843333</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>4.0</td>
      <td>...</td>
      <td>809.98</td>
      <td>33.053114</td>
      <td>274.0</td>
      <td>No</td>
      <td>49.574949</td>
      <td>21.465380</td>
      <td>High_spent_Medium_value_payments</td>
      <td>361.444004</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CUS0xd40</td>
      <td>November</td>
      <td>24</td>
      <td>Scientist</td>
      <td>19114.12</td>
      <td>1824.843333</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>4.0</td>
      <td>...</td>
      <td>809.98</td>
      <td>33.811894</td>
      <td>270.0</td>
      <td>No</td>
      <td>49.574949</td>
      <td>148.233938</td>
      <td>Low_spent_Medium_value_payments</td>
      <td>264.675446</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CUS0xd40</td>
      <td>December</td>
      <td>24</td>
      <td>Scientist</td>
      <td>19114.12</td>
      <td>1824.843333</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>4.0</td>
      <td>...</td>
      <td>809.98</td>
      <td>32.430559</td>
      <td>276.0</td>
      <td>No</td>
      <td>49.574949</td>
      <td>39.082511</td>
      <td>High_spent_Medium_value_payments</td>
      <td>343.826873</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CUS0x21b1</td>
      <td>September</td>
      <td>28</td>
      <td>Teacher</td>
      <td>34847.84</td>
      <td>3037.986667</td>
      <td>2</td>
      <td>4</td>
      <td>6</td>
      <td>1.0</td>
      <td>...</td>
      <td>605.03</td>
      <td>25.926822</td>
      <td>327.0</td>
      <td>No</td>
      <td>18.816215</td>
      <td>39.684018</td>
      <td>High_spent_Large_value_payments</td>
      <td>485.298434</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49995</th>
      <td>CUS0x8600</td>
      <td>December</td>
      <td>4975</td>
      <td>Architect</td>
      <td>20002.88</td>
      <td>1929.906667</td>
      <td>10</td>
      <td>8</td>
      <td>29</td>
      <td>5.0</td>
      <td>...</td>
      <td>3571.70</td>
      <td>34.780553</td>
      <td>72.5</td>
      <td>Yes</td>
      <td>60.964772</td>
      <td>146.486325</td>
      <td>Low_spent_Small_value_payments</td>
      <td>275.539570</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>49996</th>
      <td>CUS0x942c</td>
      <td>September</td>
      <td>25</td>
      <td>Mechanic</td>
      <td>39628.99</td>
      <td>3359.415833</td>
      <td>4</td>
      <td>6</td>
      <td>7</td>
      <td>2.0</td>
      <td>...</td>
      <td>502.38</td>
      <td>27.758522</td>
      <td>383.0</td>
      <td>No</td>
      <td>35.104023</td>
      <td>181.442999</td>
      <td>Low_spent_Small_value_payments</td>
      <td>409.394562</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>49997</th>
      <td>CUS0x942c</td>
      <td>October</td>
      <td>25</td>
      <td>Mechanic</td>
      <td>39628.99</td>
      <td>3359.415833</td>
      <td>4</td>
      <td>6</td>
      <td>7</td>
      <td>2.0</td>
      <td>...</td>
      <td>502.38</td>
      <td>36.858542</td>
      <td>384.0</td>
      <td>No</td>
      <td>35.104023</td>
      <td>10000.000000</td>
      <td>Low_spent_Large_value_payments</td>
      <td>349.726332</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>49998</th>
      <td>CUS0x942c</td>
      <td>November</td>
      <td>25</td>
      <td>Mechanic</td>
      <td>39628.99</td>
      <td>3359.415833</td>
      <td>4</td>
      <td>6</td>
      <td>7</td>
      <td>2.0</td>
      <td>...</td>
      <td>502.38</td>
      <td>39.139840</td>
      <td>385.0</td>
      <td>No</td>
      <td>35.104023</td>
      <td>97.598580</td>
      <td>High_spent_Small_value_payments</td>
      <td>463.238981</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
    <tr>
      <th>49999</th>
      <td>CUS0x942c</td>
      <td>December</td>
      <td>25</td>
      <td>Mechanic</td>
      <td>39628.99</td>
      <td>3359.415833</td>
      <td>4</td>
      <td>6</td>
      <td>7</td>
      <td>2.0</td>
      <td>...</td>
      <td>502.38</td>
      <td>34.108530</td>
      <td>386.0</td>
      <td>No</td>
      <td>35.104023</td>
      <td>220.457878</td>
      <td>Low_spent_Medium_value_payments</td>
      <td>360.379683</td>
      <td>NaN</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>50000 rows × 26 columns</p>
</div>




```python
class ClipOutliersTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 lower_quantile,
                 upper_quantile,
                 multiply_by=1.5,
                 replace_with_median: bool = False):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.multiply_by = multiply_by
        self.replace_with_median = replace_with_median

        self.lower_limit = 0
        self.upper_limit = 0
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        q1, q3 = np.quantile(X, [self.lower_quantile, self.upper_quantile])
        iqr = q3 - q1
        self.lower_limit = q1 - (self.multiply_by * iqr)
        self.upper_limit = q3 + (self.multiply_by * iqr)
        return self

    def transform(self, X):
        if self.replace_with_median:
            return np.where(
                ((X >= self.lower_limit) & (X <= self.upper_limit)), X,
                np.median(X))
        else:
            return np.clip(X, self.lower_limit, self.upper_limit)


def get_skewness(df, lower=None, upper=None):
    columns = df.columns
    skewness: pd.Series = df[columns].skew()
    highly_skewed = skewness[(skewness <= lower) |
                             (skewness >= upper)].index.to_list()
    lowly_skewed = skewness[(skewness > lower)
                            & (skewness < upper)].index.to_list()
    return (highly_skewed, lowly_skewed)


def remove_outliers(df: pd.DataFrame):
    category = df.select_dtypes(exclude="number").columns.drop(
        ["Credit_Score", "is_train"])
    numbers = df.select_dtypes(include="number").columns

    highly_skewed, lowly_skewed = get_skewness(df[numbers],
                                               lower=-0.8,
                                               upper=0.8)

    df[highly_skewed] = df[highly_skewed].apply(
        lambda x: ClipOutliersTransformer(
            0.25, 0.75, multiply_by=1.5, replace_with_median=True).
        fit_transform(x))

    df[lowly_skewed] = df[lowly_skewed].apply(
        lambda x: ClipOutliersTransformer(
            0.25, 0.75, multiply_by=1.5, replace_with_median=False).
        fit_transform(x))
    return df
```


```python
new_df = remove_outliers(new_df)
```


```python
new_df.to_csv("new.csv", index=False)
```

# Data Visualization


```python
# Visualization Super Class
def make_boxplot(df, column, ax):
    sns.boxplot(x="Credit_Score", y=column, data=df, ax=ax, width=0.8, palette="Set2")
    plt.xticks(rotation=90)
    # add the five number summary to the plot
    plt.title(column, fontdict={"fontsize": 15})
    plt.xticks(rotation=0)

def plot_boxplot_num_cols(df):
    fig = plt.figure(figsize=(18, 18), dpi=300)
    numb_columns = df.select_dtypes(include="number").columns
    for column in numb_columns:
        ax = fig.add_subplot(5, 4, list(numb_columns).index(column)+1)
        make_boxplot(df, column, ax)
        plt.tight_layout(pad=0.3)
    plt.tight_layout()
    plt.show()

# Define function to create histograms for all numeric features
def plot_histograms(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    num_cols = len(numeric_cols)
    num_plots_per_row = 3
    num_rows = (num_cols // num_plots_per_row) + (num_cols % num_plots_per_row > 0)
    
    fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], bins=20, kde=True, ax=axes[i])
        axes[i].set_title(f'Histogram of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        
        if i >= num_cols - 1:
            for j in range(i + 1, len(axes)):
                axes[j].remove()
            break
    
    plt.tight_layout()
    plt.show()

# Define function to create Distribution plot for all numeric features
def plot_distribution_plots(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    num_cols = len(numeric_cols)
    num_plots_per_row = 3
    num_rows = (num_cols // num_plots_per_row) + (num_cols % num_plots_per_row > 0)
    
    fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(numeric_cols):
        sns.distplot(df[col], bins=20, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Density')
        
        if i >= num_cols - 1:
            for j in range(i + 1, len(axes)):
                axes[j].remove()
            break
    
    plt.tight_layout()
    plt.show()
    
def make_countplot(df: pd.DataFrame):
    cat_cols = df.select_dtypes(exclude="number").columns.drop(
        ['Credit_Score','Customer_ID', "Type_of_Loan"])
    cat_cols = list(cat_cols)
    cat_cols.pop(-1)
    cat_cols.insert(-2, "Payment_Behaviour")

    fig, axes = plt.subplots(figsize=(12, 12), dpi=300)
    fig.suptitle("Counts of categorical columns")
    axes.grid(visible=False)
    axes.xaxis.set_tick_params(labelbottom=False)
    axes.yaxis.set_tick_params(labelleft=False)

    def __plot_graph(df, col, ax: plt.Axes, legend=False):
        sns.countplot(
            data=df,
            x=col,
            ax=ax,
            hue="Credit_Score",
        )
        # label =ax.get_xlabel()
        ax.set_xlabel(col, fontdict={"size": 9})
        ax.set_title(f"by {col}", fontdict={"size": 9})
        ax.get_xticklabels()
        ax.tick_params(labelsize=7, axis="y")
        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=90,
                           fontdict=dict(size=7))
        ax.grid(False)
        if legend:
            ax.legend(shadow=True,
                      loc="best",
                      facecolor="inherit",
                      frameon=True)
        else:
            ax.legend_ = None
        plt.tight_layout(w_pad=1)

    for i, col in enumerate(cat_cols, 1):
        if i == 3:
            continue
        ax = fig.add_subplot(2, 3, i)
        __plot_graph(df, col=col, ax=ax)

    ax2 = fig.add_axes((0.74, 0.527, 0.23, 0.35))
    __plot_graph(df, col="Payment_Behaviour", ax=ax2, legend=True)
    plt.show(True)

def plot_correlation_matrix(data):
    corr = data.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig = plt.figure(figsize=(10, 10), dpi=150)

    sns.heatmap(corr, annot=True, mask=mask, fmt=".0%", annot_kws={"size":10})
    plt.grid(False)
    plt.tick_params(axis="both", labelsize=5)
    plt.tight_layout()
    plt.title("Correlation Matrix")
    plt.show()
```


```python
plot_histograms(new_df)
```


    
![png](![Track2 - Duleep_23_0](https://github.com/duleepdaniel/creditScoreEvaluation/assets/67256324/2a25bf59-ee11-4d12-8d24-e4299606b43b))

    



```python
plot_boxplot_num_cols(new_df)
```
![Track2 - Duleep_24_0](https://github.com/duleepdaniel/creditScoreEvaluation/assets/67256324/15c8f102-720c-4898-b2d9-2a086ddda2e4)




```python
plot_distribution_plots(new_df)
```


    
![Track2 - Duleep_25_0](https://github.com/duleepdaniel/creditScoreEvaluation/assets/67256324/3824fa8b-82eb-4f26-8613-916ef0624b7f)




```python
make_countplot(new_df)
```


    
![Track2 - Duleep_26_0](https://github.com/duleepdaniel/creditScoreEvaluation/assets/67256324/3ba54595-e3eb-435d-a84c-5b83570ba68f)

    



```python
plot_correlation_matrix(new_df)
```


    
![Track2 - Duleep_27_0](https://github.com/duleepdaniel/creditScoreEvaluation/assets/67256324/99284d30-73ca-4838-bfb3-383fe3dda9dd)

    


# Feature Engineering


```python
def feature_engineering(df):
    # Define mappings for categorical variables

    credit_score_mapping = {
        "Poor": 0,
        "Standard": 1,
        "Good": 2
    }

    credit_mix_mapping = {
        "Bad": 0,
        "Standard": 1,
        "Good": 2
    }

    min_amount_mapping = {
        "Yes": 1,
        "No": 0
    }

    # Replace categorical variables with mapped values

    df['Credit_Score'].replace(credit_score_mapping, inplace=True)
    df['Credit_Mix'].replace(credit_mix_mapping, inplace=True)
    df['Payment_of_Min_Amount'].replace(min_amount_mapping, inplace=True)

    # Perform one-hot encoding for selected categorical variables

    df = pd.get_dummies(df, columns=['Occupation', 'Payment_Behaviour'])

    # Drop unnecessary columns

    df = df.drop(['Customer_ID', 'Month', 'Type_of_Loan', 'is_train'], axis=1)

    return df
```


```python
df = new_df[new_df["is_train"]]
```


```python
df = feature_engineering(df)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 100000 entries, 0 to 99999
    Data columns (total 41 columns):
     #   Column                                              Non-Null Count   Dtype  
    ---  ------                                              --------------   -----  
     0   Age                                                 100000 non-null  float64
     1   Annual_Income                                       100000 non-null  float64
     2   Monthly_Inhand_Salary                               100000 non-null  float64
     3   Num_Bank_Accounts                                   100000 non-null  float64
     4   Num_Credit_Card                                     100000 non-null  float64
     5   Interest_Rate                                       100000 non-null  float64
     6   Num_of_Loan                                         100000 non-null  float64
     7   Delay_from_due_date                                 100000 non-null  float64
     8   Num_of_Delayed_Payment                              100000 non-null  float64
     9   Changed_Credit_Limit                                100000 non-null  float64
     10  Num_Credit_Inquiries                                100000 non-null  float64
     11  Credit_Mix                                          100000 non-null  int64  
     12  Outstanding_Debt                                    100000 non-null  float64
     13  Credit_Utilization_Ratio                            100000 non-null  float64
     14  Credit_History_Age                                  100000 non-null  float64
     15  Payment_of_Min_Amount                               100000 non-null  int64  
     16  Total_EMI_per_month                                 100000 non-null  float64
     17  Amount_invested_monthly                             100000 non-null  float64
     18  Monthly_Balance                                     100000 non-null  float64
     19  Credit_Score                                        100000 non-null  int64  
     20  Occupation_Accountant                               100000 non-null  uint8  
     21  Occupation_Architect                                100000 non-null  uint8  
     22  Occupation_Developer                                100000 non-null  uint8  
     23  Occupation_Doctor                                   100000 non-null  uint8  
     24  Occupation_Engineer                                 100000 non-null  uint8  
     25  Occupation_Entrepreneur                             100000 non-null  uint8  
     26  Occupation_Journalist                               100000 non-null  uint8  
     27  Occupation_Lawyer                                   100000 non-null  uint8  
     28  Occupation_Manager                                  100000 non-null  uint8  
     29  Occupation_Mechanic                                 100000 non-null  uint8  
     30  Occupation_MediaManager                             100000 non-null  uint8  
     31  Occupation_Musician                                 100000 non-null  uint8  
     32  Occupation_Scientist                                100000 non-null  uint8  
     33  Occupation_Teacher                                  100000 non-null  uint8  
     34  Occupation_Writer                                   100000 non-null  uint8  
     35  Payment_Behaviour_High_spent_Large_value_payments   100000 non-null  uint8  
     36  Payment_Behaviour_High_spent_Medium_value_payments  100000 non-null  uint8  
     37  Payment_Behaviour_High_spent_Small_value_payments   100000 non-null  uint8  
     38  Payment_Behaviour_Low_spent_Large_value_payments    100000 non-null  uint8  
     39  Payment_Behaviour_Low_spent_Medium_value_payments   100000 non-null  uint8  
     40  Payment_Behaviour_Low_spent_Small_value_payments    100000 non-null  uint8  
    dtypes: float64(17), int64(3), uint8(21)
    memory usage: 18.0 MB


# Modeling and Evaluation


```python
X, y = df.drop("Credit_Score",axis=1).values , df["Credit_Score"]
```


```python
y.value_counts(normalize=True)
```




    1    0.53174
    0    0.28998
    2    0.17828
    Name: Credit_Score, dtype: float64




```python
rus = SMOTE(sampling_strategy='auto')
X_data_rus, y_data_rus = rus.fit_resample(X, y)
```


```python
y_data_rus.value_counts(normalize=True)
```




    2    0.333333
    1    0.333333
    0    0.333333
    Name: Credit_Score, dtype: float64




```python
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data_rus, y_data_rus, test_size=0.3, random_state=42,stratify=y_data_rus)
```


```python
scalar = PowerTransformer(method='yeo-johnson', standardize=True).fit(X_train)
```


```python
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)
```


```python
# Define the models
models = {
    "Bagging": BaggingClassifier(n_jobs=-1),
    "ExtraTrees": ExtraTreesClassifier(max_depth=10, n_jobs=-1),
    "RandomForest": RandomForestClassifier(n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingClassifier(),
    "XGB": XGBClassifier(n_jobs=-1),
    "KNN": KNeighborsClassifier(),
    "AdaBoost": AdaBoostClassifier()
}
```


```python
# Initialize dictionaries to store scores
precision_scores = {}
recall_scores = {}
f1_scores = {}

# Iterate over each model
for model_name, model in models.items():
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred = model.predict(X_test)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report)

    # Store scores in dictionaries
    precision_scores[model_name] = report['weighted avg']['precision']
    recall_scores[model_name] = report['weighted avg']['recall']
    f1_scores[model_name] = report['weighted avg']['f1-score']
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print the classification report for the model
    print(f"{model_name} Classification Report:")
    print(report_df)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=True, yticklabels=True)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

```

    Bagging Classification Report:
                          0             1             2  accuracy     macro avg  \
    precision      0.831455      0.798765      0.859066  0.830662      0.829762   
    recall         0.869609      0.746239      0.876136  0.830662      0.830661   
    f1-score       0.850104      0.771609      0.867517  0.830662      0.829743   
    support    15952.000000  15952.000000  15953.000000  0.830662  47857.000000   
    
               weighted avg  
    precision      0.829763  
    recall         0.830662  
    f1-score       0.829744  
    support    47857.000000  



    
![Track2 - Duleep_42_1](https://github.com/duleepdaniel/creditScoreEvaluation/assets/67256324/da12dcef-8b60-44a3-8abb-bc7c6f2e8c97)

    


    ExtraTrees Classification Report:
                          0             1             2  accuracy     macro avg  \
    precision      0.755234      0.744669      0.699620  0.729569      0.733174   
    recall         0.741788      0.582309      0.864602  0.729569      0.729567   
    f1-score       0.748450      0.653557      0.773410  0.729569      0.725139   
    support    15952.000000  15952.000000  15953.000000  0.729569  47857.000000   
    
               weighted avg  
    precision      0.733173  
    recall         0.729569  
    f1-score       0.725140  
    support    47857.000000  



    
![Track2 - Duleep_42_3](https://github.com/duleepdaniel/creditScoreEvaluation/assets/67256324/3abd9e22-d44c-4b3b-9a14-928b601219d7)

    


    RandomForest Classification Report:
                          0             1             2  accuracy     macro avg  \
    precision      0.864862      0.847089      0.851255  0.854609      0.854402   
    recall         0.877006      0.759842      0.926973  0.854609      0.854607   
    f1-score       0.870891      0.801097      0.887502  0.854609      0.853163   
    support    15952.000000  15952.000000  15953.000000  0.854609  47857.000000   
    
               weighted avg  
    precision      0.854402  
    recall         0.854609  
    f1-score       0.853164  
    support    47857.000000  



    
![Track2 - Duleep_42_5](https://github.com/duleepdaniel/creditScoreEvaluation/assets/67256324/f922d59d-ad93-4cd1-9cc8-91d73abbcfc9)

    


    HistGradientBoosting Classification Report:
                          0             1             2  accuracy     macro avg  \
    precision      0.838349      0.790887      0.806791  0.811919      0.812009   
    recall         0.798144      0.729062      0.908544  0.811919      0.811917   
    f1-score       0.817753      0.758717      0.854649  0.811919      0.810373   
    support    15952.000000  15952.000000  15953.000000  0.811919  47857.000000   
    
               weighted avg  
    precision      0.812009  
    recall         0.811919  
    f1-score       0.810374  
    support    47857.000000  



    
![Track2 - Duleep_42_7](https://github.com/duleepdaniel/creditScoreEvaluation/assets/67256324/637f2603-dfd0-48d2-8f75-5db0aa38d9ae)

    


    XGB Classification Report:
                          0             1             2  accuracy     macro avg  \
    precision      0.850976      0.800133      0.838602  0.830558      0.829904   
    recall         0.827984      0.752382      0.911302  0.830558      0.830556   
    f1-score       0.839323      0.775523      0.873442  0.830558      0.829429   
    support    15952.000000  15952.000000  15953.000000  0.830558  47857.000000   
    
               weighted avg  
    precision      0.829904  
    recall         0.830558  
    f1-score       0.829430  
    support    47857.000000  



    
![Track2 - Duleep_42_9](https://github.com/duleepdaniel/creditScoreEvaluation/assets/67256324/8ddc27b2-eac7-4418-a317-75a54a1d4275)

    


    KNN Classification Report:
                          0             1             2  accuracy     macro avg  \
    precision      0.782169      0.761925      0.827951  0.792402      0.790682   
    recall         0.828799      0.670888      0.877515  0.792402      0.792401   
    f1-score       0.804809      0.713514      0.852013  0.792402      0.790112   
    support    15952.000000  15952.000000  15953.000000  0.792402  47857.000000   
    
               weighted avg  
    precision      0.790683  
    recall         0.792402  
    f1-score       0.790113  
    support    47857.000000  



    
![Track2 - Duleep_42_11](https://github.com/duleepdaniel/creditScoreEvaluation/assets/67256324/9e301952-93fa-40e2-b56b-46853a43877d)

    


    AdaBoost Classification Report:
                          0             1             2  accuracy     macro avg  \
    precision      0.750785      0.751086      0.707051  0.732516      0.736307   
    recall         0.689882      0.617603      0.890052  0.732516      0.732512   
    f1-score       0.719046      0.677835      0.788067  0.732516      0.728316   
    support    15952.000000  15952.000000  15953.000000  0.732516  47857.000000   
    
               weighted avg  
    precision      0.736307  
    recall         0.732516  
    f1-score       0.728317  
    support    47857.000000  



    
![Track2 - Duleep_42_13](https://github.com/duleepdaniel/creditScoreEvaluation/assets/67256324/741f0897-db97-4d8e-a2a7-d84aa60bb523)

    



```python
# Comparing Models
plt.figure(figsize=(12, 6))

# Precision Scores
plt.subplot(1, 3, 1)
sns.barplot(x=list(precision_scores.keys()), y=list(precision_scores.values()))
plt.title("Precision Scores")
plt.xticks(rotation=90)

# Recall Scores
plt.subplot(1, 3, 2)
sns.barplot(x=list(recall_scores.keys()), y=list(recall_scores.values()))
plt.title("Recall Scores")
plt.xticks(rotation=90)

# F1 Scores
plt.subplot(1, 3, 3)
sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()))
plt.title("F1 Scores")
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()
```


    
![Track2 - Duleep_43_0](https://github.com/duleepdaniel/creditScoreEvaluation/assets/67256324/a1c09e6d-ed9e-4d15-8159-1ccc78967f30)

    


## StackingClassifier


```python
model = StackingClassifier(list(models.items()), n_jobs=-1)
```


```python
model.fit(X_train, y_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>StackingClassifier(estimators=[(&#x27;Bagging&#x27;, BaggingClassifier(n_jobs=-1)),
                               (&#x27;ExtraTrees&#x27;,
                                ExtraTreesClassifier(max_depth=10, n_jobs=-1)),
                               (&#x27;RandomForest&#x27;,
                                RandomForestClassifier(n_jobs=-1)),
                               (&#x27;HistGradientBoosting&#x27;,
                                HistGradientBoostingClassifier()),
                               (&#x27;XGB&#x27;,
                                XGBClassifier(base_score=None, booster=None,
                                              callbacks=None,
                                              colsample_bylevel=None,
                                              colsample_bynode=None...
                                              learning_rate=None, max_bin=None,
                                              max_cat_threshold=None,
                                              max_cat_to_onehot=None,
                                              max_delta_step=None,
                                              max_depth=None, max_leaves=None,
                                              min_child_weight=None,
                                              missing=nan,
                                              monotone_constraints=None,
                                              multi_strategy=None,
                                              n_estimators=None, n_jobs=-1,
                                              num_parallel_tree=None,
                                              objective=&#x27;multi:softprob&#x27;, ...)),
                               (&#x27;KNN&#x27;, KNeighborsClassifier()),
                               (&#x27;AdaBoost&#x27;, AdaBoostClassifier())],
                   n_jobs=-1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">StackingClassifier</label><div class="sk-toggleable__content"><pre>StackingClassifier(estimators=[(&#x27;Bagging&#x27;, BaggingClassifier(n_jobs=-1)),
                               (&#x27;ExtraTrees&#x27;,
                                ExtraTreesClassifier(max_depth=10, n_jobs=-1)),
                               (&#x27;RandomForest&#x27;,
                                RandomForestClassifier(n_jobs=-1)),
                               (&#x27;HistGradientBoosting&#x27;,
                                HistGradientBoostingClassifier()),
                               (&#x27;XGB&#x27;,
                                XGBClassifier(base_score=None, booster=None,
                                              callbacks=None,
                                              colsample_bylevel=None,
                                              colsample_bynode=None...
                                              learning_rate=None, max_bin=None,
                                              max_cat_threshold=None,
                                              max_cat_to_onehot=None,
                                              max_delta_step=None,
                                              max_depth=None, max_leaves=None,
                                              min_child_weight=None,
                                              missing=nan,
                                              monotone_constraints=None,
                                              multi_strategy=None,
                                              n_estimators=None, n_jobs=-1,
                                              num_parallel_tree=None,
                                              objective=&#x27;multi:softprob&#x27;, ...)),
                               (&#x27;KNN&#x27;, KNeighborsClassifier()),
                               (&#x27;AdaBoost&#x27;, AdaBoostClassifier())],
                   n_jobs=-1)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>Bagging</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">BaggingClassifier</label><div class="sk-toggleable__content"><pre>BaggingClassifier(n_jobs=-1)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>ExtraTrees</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">ExtraTreesClassifier</label><div class="sk-toggleable__content"><pre>ExtraTreesClassifier(max_depth=10, n_jobs=-1)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>RandomForest</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(n_jobs=-1)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>HistGradientBoosting</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">HistGradientBoostingClassifier</label><div class="sk-toggleable__content"><pre>HistGradientBoostingClassifier()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>XGB</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">XGBClassifier</label><div class="sk-toggleable__content"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              multi_strategy=None, n_estimators=None, n_jobs=-1,
              num_parallel_tree=None, objective=&#x27;multi:softprob&#x27;, ...)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>KNN</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsClassifier</label><div class="sk-toggleable__content"><pre>KNeighborsClassifier()</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>AdaBoost</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">AdaBoostClassifier</label><div class="sk-toggleable__content"><pre>AdaBoostClassifier()</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>final_estimator</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div></div></div></div></div></div></div></div>




```python
print("Train Score: ",model.score(X_train, y_train))
```

    Train Score:  0.9987552053015717



```python
print("Test Score: ",model.score(X_test, y_test))
```

    Test Score:  0.8664145266105272



```python
y_pred = model.predict(X_test)
```


```python
print(classification_report(y_pred,y_test))
```

                  precision    recall  f1-score   support
    
               0       0.88      0.88      0.88     15990
               1       0.80      0.82      0.81     15658
               2       0.92      0.90      0.91     16209
    
        accuracy                           0.87     47857
       macro avg       0.87      0.87      0.87     47857
    weighted avg       0.87      0.87      0.87     47857
    



```python
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=True, yticklabels=True)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Final Model Confusion Matrix')
    plt.show()
```


    
![Track2 - Duleep_51_0](https://github.com/duleepdaniel/creditScoreEvaluation/assets/67256324/17ff18ab-769b-4954-bfd3-15ce5ca881e8)

    


## Feature Importance


```python
# Permutation Importances for HistGradientBoostingClassifier
models["HistGradientBoosting"].fit(X_train, y_train)
result = permutation_importance(models["HistGradientBoosting"], X_test, y_test, n_repeats=10, random_state=42)
```


```python
# Get feature importances
hist_gb_feature_importance = result.importances_mean
```


```python
# Feature names
feature_names = df.drop("Credit_Score", axis=1).columns

# Sort feature importances and feature names together
sorted_indices = np.argsort(hist_gb_feature_importance)
sorted_feature_names = feature_names[sorted_indices]
sorted_importances = hist_gb_feature_importance[sorted_indices]

# Choose colors for the bars
colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_feature_names)))

# Plot permutation importances with colors and sorted order
plt.figure(figsize=(10, 15))
plt.barh(sorted_feature_names, sorted_importances, color=colors)
plt.xlabel('Mean Permutation Importance')
plt.ylabel('Feature')
plt.title('Importances of Features')
plt.show()
```


    
![Track2 - Duleep_55_0](https://github.com/duleepdaniel/creditScoreEvaluation/assets/67256324/cf627d56-901d-4e1a-8de8-eb69c8b3c529)

    



```python
cross_tab = pd.crosstab(values=df["Monthly_Balance"], index=[
                        df["Credit_Score"], df["Credit_Mix"]], columns="Monthly_Balance", aggfunc="mean").reset_index()

main_group = pd.pivot_table(cross_tab, "Monthly_Balance", "Credit_Score", aggfunc=np.mean)
cross_tab
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>Credit_Score</th>
      <th>Credit_Mix</th>
      <th>Monthly_Balance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>287.496216</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>330.466093</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>402.154168</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>278.749082</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>375.520867</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>2</td>
      <td>396.714662</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>0</td>
      <td>284.419754</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>1</td>
      <td>375.099900</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>2</td>
      <td>397.418036</td>
    </tr>
  </tbody>
</table>
</div>




```python
a = plt.cm.Accent
b = plt.cm.Blues

fig, ax = plt.subplots(figsize=(6, 4))
fig.suptitle("Distribution of Monthly_Balance by Credit Score & Credit Mix",
             fontsize=11,
             color="k")
fig.set_frameon(True)

pie1, *_, texts = ax.pie(x=main_group["Monthly_Balance"],
                         labels=main_group.index,
                         autopct="%.1f%%",
                         radius=1.3,
                         colors=[a(120, 1), b(100, 1),
                                 a(0, 1)],
                         pctdistance=0.8,
                         textprops={"size": 9},
                         frame=True)
plt.setp(pie1, width=0.5)
ax.set_frame_on(True)

pie2, *_, texts = ax.pie(x=cross_tab["Monthly_Balance"],
                         autopct="%.0f%%",
                         radius=0.8,
                         colors=[
                             a(80, 0.9),
                             a(80, 0.8),
                             a(80, 0.7),
                             b(100, 0.9),
                             b(100, 0.8),
                             b(100, 0.7),
                             a(0, 0.8),
                             a(0, 0.65),
                             a(0, 0.5)
],
    textprops={"size": 8})
plt.setp(pie2, width=0.5)
legend_labels = np.unique(cross_tab["Credit_Mix"])

legend_handles = [
    plt.plot([], label=legend_labels[0], c="k"),
    plt.plot([], label=legend_labels[1], c='b'),
    plt.plot([], label=legend_labels[-1], c="g")
]
plt.legend(shadow=True,
           frameon=True,
           facecolor="inherit",
           loc="best",
           title="credit Score & Mix",
           bbox_to_anchor=(1, 1, 0.5, 0.1))

plt.show()
```


    
![Track2 - Duleep_57_0](https://github.com/duleepdaniel/creditScoreEvaluation/assets/67256324/2a4e7531-59eb-4ed2-9f3f-1104285d42b3)

    


# Predicting Test Data


```python
df_test = new_df[~new_df["is_train"]]
```


```python
df_test = feature_engineering(df_test)
```


```python
df_test.drop(["Credit_Score"], axis=1, inplace=True, errors="ignore")
```


```python
df_test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 50000 entries, 0 to 49999
    Data columns (total 40 columns):
     #   Column                                              Non-Null Count  Dtype  
    ---  ------                                              --------------  -----  
     0   Age                                                 50000 non-null  float64
     1   Annual_Income                                       50000 non-null  float64
     2   Monthly_Inhand_Salary                               50000 non-null  float64
     3   Num_Bank_Accounts                                   50000 non-null  float64
     4   Num_Credit_Card                                     50000 non-null  float64
     5   Interest_Rate                                       50000 non-null  float64
     6   Num_of_Loan                                         50000 non-null  float64
     7   Delay_from_due_date                                 50000 non-null  float64
     8   Num_of_Delayed_Payment                              50000 non-null  float64
     9   Changed_Credit_Limit                                50000 non-null  float64
     10  Num_Credit_Inquiries                                50000 non-null  float64
     11  Credit_Mix                                          50000 non-null  int64  
     12  Outstanding_Debt                                    50000 non-null  float64
     13  Credit_Utilization_Ratio                            50000 non-null  float64
     14  Credit_History_Age                                  50000 non-null  float64
     15  Payment_of_Min_Amount                               50000 non-null  int64  
     16  Total_EMI_per_month                                 50000 non-null  float64
     17  Amount_invested_monthly                             50000 non-null  float64
     18  Monthly_Balance                                     50000 non-null  float64
     19  Occupation_Accountant                               50000 non-null  uint8  
     20  Occupation_Architect                                50000 non-null  uint8  
     21  Occupation_Developer                                50000 non-null  uint8  
     22  Occupation_Doctor                                   50000 non-null  uint8  
     23  Occupation_Engineer                                 50000 non-null  uint8  
     24  Occupation_Entrepreneur                             50000 non-null  uint8  
     25  Occupation_Journalist                               50000 non-null  uint8  
     26  Occupation_Lawyer                                   50000 non-null  uint8  
     27  Occupation_Manager                                  50000 non-null  uint8  
     28  Occupation_Mechanic                                 50000 non-null  uint8  
     29  Occupation_MediaManager                             50000 non-null  uint8  
     30  Occupation_Musician                                 50000 non-null  uint8  
     31  Occupation_Scientist                                50000 non-null  uint8  
     32  Occupation_Teacher                                  50000 non-null  uint8  
     33  Occupation_Writer                                   50000 non-null  uint8  
     34  Payment_Behaviour_High_spent_Large_value_payments   50000 non-null  uint8  
     35  Payment_Behaviour_High_spent_Medium_value_payments  50000 non-null  uint8  
     36  Payment_Behaviour_High_spent_Small_value_payments   50000 non-null  uint8  
     37  Payment_Behaviour_Low_spent_Large_value_payments    50000 non-null  uint8  
     38  Payment_Behaviour_Low_spent_Medium_value_payments   50000 non-null  uint8  
     39  Payment_Behaviour_Low_spent_Small_value_payments    50000 non-null  uint8  
    dtypes: float64(17), int64(2), uint8(21)
    memory usage: 8.6 MB



```python
# Transform the test data using the same scalar used for training
X_test_processed = scalar.transform(df_test.values)
```


```python
# Predict Credit_Score for the test data
y_pred_test = model.predict(X_test_processed)
```


```python
# Add the predicted Credit_Score to the test dataframe
df_test['Predicted_Credit_Score'] = y_pred_test
```


```python
df_test.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Annual_Income</th>
      <th>Monthly_Inhand_Salary</th>
      <th>Num_Bank_Accounts</th>
      <th>Num_Credit_Card</th>
      <th>Interest_Rate</th>
      <th>Num_of_Loan</th>
      <th>Delay_from_due_date</th>
      <th>Num_of_Delayed_Payment</th>
      <th>Changed_Credit_Limit</th>
      <th>...</th>
      <th>Occupation_Scientist</th>
      <th>Occupation_Teacher</th>
      <th>Occupation_Writer</th>
      <th>Payment_Behaviour_High_spent_Large_value_payments</th>
      <th>Payment_Behaviour_High_spent_Medium_value_payments</th>
      <th>Payment_Behaviour_High_spent_Small_value_payments</th>
      <th>Payment_Behaviour_Low_spent_Large_value_payments</th>
      <th>Payment_Behaviour_Low_spent_Medium_value_payments</th>
      <th>Payment_Behaviour_Low_spent_Small_value_payments</th>
      <th>Predicted_Credit_Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23.0</td>
      <td>19114.12</td>
      <td>1824.843333</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>11.27</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24.0</td>
      <td>19114.12</td>
      <td>1824.843333</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>13.27</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>24.0</td>
      <td>19114.12</td>
      <td>1824.843333</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>12.27</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24.0</td>
      <td>19114.12</td>
      <td>1824.843333</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>11.27</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28.0</td>
      <td>34847.84</td>
      <td>3037.986667</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>5.42</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>28.0</td>
      <td>34847.84</td>
      <td>3037.986667</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>5.42</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>28.0</td>
      <td>34847.84</td>
      <td>3037.986667</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.5</td>
      <td>5.42</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>28.0</td>
      <td>34847.84</td>
      <td>3037.986667</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>7.42</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>35.0</td>
      <td>143162.64</td>
      <td>12187.220000</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>14.0</td>
      <td>7.10</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>35.0</td>
      <td>143162.64</td>
      <td>12187.220000</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>2.10</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 41 columns</p>
</div>




```python
df_test.to_csv('df_test_predicted.csv', index=False)
```


```python

```
