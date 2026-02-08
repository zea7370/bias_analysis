import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(path):
    df = pd.read_csv(path)

    # Remove missing values
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Target variable
    df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

    # Sensitive attributes
    sensitive_features = df[['sex', 'race']]

    # Encode categorical features
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    X = df_encoded.drop('income', axis=1)
    y = df_encoded['income']

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive_features, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test, sens_train, sens_test
