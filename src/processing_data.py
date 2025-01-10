import pandas as pd
from sklearn.preprocessing import MinMaxScaler

column_names = ["kepid", "kepoi_name", "kepler_name", "koi_disposition", "koi_pdisposition", "koi_score",
                "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec", "koi_period",
                "koi_period_err1", "koi_period_err2", "koi_time0bk", "koi_time0bk_err1", "koi_time0bk_err2",
                "koi_impact", "koi_impact_err1", "koi_impact_err2", "koi_duration", "koi_duration_err1",
                "koi_duration_err2", "koi_depth", "koi_depth_err1", "koi_depth_err2", "koi_prad",
                "koi_prad_err1", "koi_prad_err2", "koi_teq", "koi_teq_err1", "koi_teq_err2", "koi_insol",
                "koi_insol_err1", "koi_insol_err2", "koi_model_snr", "koi_tce_plnt_num", "koi_tce_delivname",
                "koi_steff", "koi_steff_err1", "koi_steff_err2", "koi_slogg", "koi_slogg_err1", "koi_slogg_err2",
                "koi_srad", "koi_srad_err1", "koi_srad_err2", "ra", "dec", "koi_kepmag"]

def load_data(filepath):
    return pd.read_csv(filepath, names=column_names, skiprows=1)

def clean_data(data):
    print("Initial data shape:", data.shape)

    # Step 1: Handle missing values
    data = data.dropna(subset=['koi_prad', 'koi_period'])  # Drop rows with critical missing values
    print("After dropping rows with missing koi_prad and koi_period:", data.shape)

    # Step 2: Filter by disposition
    data = data[data['koi_disposition'].isin(['CONFIRMED', 'CANDIDATE'])]
    print("After filtering koi_disposition:", data.shape)

    # Step 3: Filter by realistic ranges
    data = data[(data['koi_prad'].between(0.5, 15)) & (data['koi_period'].between(0.1, 365))]
    print("After filtering koi_prad and koi_period ranges:", data.shape)

    return data


def feature_engineering(data):
    if 'koi_teq' in data.columns:
        data['hz_index'] = data['koi_teq'] / 300  # Use 'koi_teq' instead of 'pl_eqt'
    else:
        raise KeyError("Column 'koi_teq' is missing in the dataset.")
    return data

def scale_features(data, features):
    scaler = MinMaxScaler()
    
    if data.empty:
        raise ValueError("Dataset is empty. Cannot scale features.")
    
    missing_features = [feature for feature in features if feature not in data.columns]
    if missing_features:
        raise KeyError(f"Missing features for scaling: {missing_features}")
    
    data[features] = scaler.fit_transform(data[features])
    return data

# Main function
if __name__ == "__main__":
    # Load data
    data = load_data('data/exoplanet_koi.csv')
    print("Loaded data shape:", data.shape)

    # Clean data
    data = clean_data(data)
    print("Data after cleaning:", data.shape)

    # Feature engineering
    data = feature_engineering(data)

    # Scale features
    scaling_features = ['koi_prad', 'koi_period']
    data = scale_features(data, scaling_features)

    # Save processed data
    data.to_csv('data/exoplanet_processed.csv', index=False)
    print("Processed data saved to 'data/exoplanet_processed.csv'")
