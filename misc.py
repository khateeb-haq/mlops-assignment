import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error


def load_data():
    """Load the Boston housing dataset from the original CMU source."""
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    # now we split this into data and target
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    # These are the Feature names based on the original dataset
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    # Create a DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target  # here MEDV is our target variable
    return df


def preprocess_data(df, target_col, test_size=0.2, random_state=42):
    """
    Split a DataFrame into train/test sets.

    Returns X_train, X_test, y_train, y_test.
    Works with any tabular dataset given a target column name.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train):
    """
    Fit any scikit-learn compatible model on training data.

    Returns the trained model.
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.

    Returns the Mean Squared Error (MSE).
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse


def average_cv_mse(model, X, y, cv=5):
    """
    Compute average MSE across k cross-validation folds.

    Works with any scikit-learn compatible regression model.
    Returns the mean MSE across all folds.
    """
    scores = cross_val_score(
        model, X, y, scoring="neg_mean_squared_error", cv=cv
    )
    return -scores.mean()
