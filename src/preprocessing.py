import pandas as pd

def clean_data(df):
    """
    Cleans the dataset by filling missing values.
    
    - Numeric skewed columns: fill with median
    - Categorical columns: fill with mode
    """
    # Fill numeric columns (right-skewed)
    col_num = ['LoanAmount']
    for feature in col_num:
        df[feature].fillna(df[feature].median(), inplace=True)

    # Fill categorical columns
    cat_col = ['Credit_History', 'Self_Employed', 'Dependents',
               'Loan_Amount_Term', 'Gender', 'Married']
    for feature in cat_col:
        df[feature].fillna(df[feature].mode()[0], inplace=True)
    
    return df
