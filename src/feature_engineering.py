def create_features(df):
    # Example: add debt-to-income ratio
    df["debt_to_income"] = df["loan_amount"] / (df["income"] + 1)
    return df
