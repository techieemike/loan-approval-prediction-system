def clean_data(df):
    col_num = ['LoanAmount']
    for feature in col_num:
        if feature in df.columns:
            df[feature].fillna(df[feature].median(), inplace=True)

    cat_col = ['Credit_History', 'Self_Employed', 'Dependents', 
               'Loan_Amount_Term', 'Gender', 'Married']
    for feature in cat_col:
        if feature in df.columns:
            df[feature].fillna(df[feature].mode()[0], inplace=True)

    return df
