import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # One-Hot Encode the 'Location' column
    df = pd.get_dummies(df, columns=['Location'], drop_first=True)

    # Split features and target
    X = df.drop('Price', axis=1)
    y = df['Price']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler
