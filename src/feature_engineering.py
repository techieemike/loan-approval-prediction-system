import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
import os

# Globals
label_encoders = {}
onehot_encoder = None
scaler = None
pca = None

BINARY_COLS = ['Gender', 'Married', 'Education', 'Self_Employed']
ONEHOT_COLS = ['Property_Area', 'Dependents']
NUMERIC_COLS = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

MODEL_DIR = "model"
SCALER_DIR = "/model/scaler"
PCA_DIR = "/model/pca"

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(PCA_DIR, exist_ok=True)

def get_onehot_feature_names(ohe, input_cols):
    # Use get_feature_names if available, else build names manually
    if hasattr(ohe, "get_feature_names"):
        return ohe.get_feature_names(input_cols)
    else:
        feature_names = []
        for i, cats in enumerate(ohe.categories_):
            feature_names.extend([f"{input_cols[i]}_{cat}" for cat in cats])
        return feature_names

def create_features(df, training=True, apply_pca=False, n_components=5):
    global label_encoders, onehot_encoder, scaler, pca

    df = df.copy()

    # Drop unwanted columns
    df.drop(columns=[col for col in ['Loan_ID', 'Loan_Status'] if col in df.columns], inplace=True)

    # Load transformers if inference
    if not training:
        label_encoders = joblib.load(f"{MODEL_DIR}/label_encoders.pkl")
        onehot_encoder = joblib.load(f"{MODEL_DIR}/onehot_encoder.pkl")
        scaler = joblib.load(f"{SCALER_DIR}/scaler.pkl")
        if apply_pca and os.path.exists(f"{PCA_DIR}/pca.pkl"):
            pca = joblib.load(f"{PCA_DIR}/pca.pkl")

    # Label encode binary categorical
    for col in BINARY_COLS:
        if training:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            df[col] = le.transform(df[col].astype(str))

    # One-hot encode multiclass categorical
    if training:
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        onehot_encoded = onehot_encoder.fit_transform(df[ONEHOT_COLS])
    else:
        onehot_encoded = onehot_encoder.transform(df[ONEHOT_COLS])

    feature_names = get_onehot_feature_names(onehot_encoder, ONEHOT_COLS)
    onehot_df = pd.DataFrame(onehot_encoded, columns=feature_names, index=df.index)
    df.drop(columns=ONEHOT_COLS, inplace=True)
    df = pd.concat([df, onehot_df], axis=1)

    # Scale numeric features
    if training:
        scaler = StandardScaler()
        df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
    else:
        df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    # PCA transformation
    if apply_pca:
        if training:
            pca = PCA(n_components=n_components)
            pcs = pca.fit_transform(df)
            joblib.dump(pca, f"{PCA_DIR}/pca.pkl")
        else:
            pcs = pca.transform(df)
        df = pd.DataFrame(pcs, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=df.index)

    # Save transformers on training
    if training:
        joblib.dump(label_encoders, f"{MODEL_DIR}/label_encoders.pkl")
        joblib.dump(onehot_encoder, f"{MODEL_DIR}/onehot_encoder.pkl")
        joblib.dump(scaler, f"{SCALER_DIR}/scaler.pkl")

    return df

def preprocess_features_for_input(data_dict):
    """Process a single input dictionary for prediction (inference only)."""
    global label_encoders, onehot_encoder, scaler, pca

    # Convert dict to DataFrame
    df = pd.DataFrame([data_dict])

    # Load transformers if not already loaded
    if not label_encoders:
        label_encoders.update(joblib.load(f"{MODEL_DIR}/label_encoders.pkl"))
    if not onehot_encoder:
        onehot_encoder = joblib.load(f"{MODEL_DIR}/onehot_encoder.pkl")
    if not scaler:
        scaler = joblib.load(f"{SCALER_DIR}/scaler.pkl")
    if os.path.exists(f"{PCA_DIR}/pca.pkl") and not pca:
        pca = joblib.load(f"{PCA_DIR}/pca.pkl")

    # Label encode binary features
    for col in BINARY_COLS:
        df[col] = label_encoders[col].transform(df[col].astype(str))

    # One-hot encode
    onehot_encoded = onehot_encoder.transform(df[ONEHOT_COLS])
    feature_names = get_onehot_feature_names(onehot_encoder, ONEHOT_COLS)
    onehot_df = pd.DataFrame(onehot_encoded, columns=feature_names, index=df.index)
    df.drop(columns=ONEHOT_COLS, inplace=True)
    df = pd.concat([df, onehot_df], axis=1)

    # Scale numeric
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    # Apply PCA if available
    if pca:
        df = pd.DataFrame(pca.transform(df), columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=df.index)

    return df
