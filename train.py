from sklearn.tree import DecisionTreeRegressor

from misc import load_data, preprocess_data, train_model, evaluate_model, average_cv_mse

TARGET_COL = "MEDV"


def main():
    # 1. Load data
    df = load_data()
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Preprocess — train/test split
    X_train, X_test, y_train, y_test = preprocess_data(df, TARGET_COL)
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # 3. Train
    model = DecisionTreeRegressor(random_state=42)
    model = train_model(model, X_train, y_train)
    print("Model training complete.")

    # 4. Evaluate on test set
    mse = evaluate_model(model, X_test, y_test)
    print(f"\nModel     : DecisionTreeRegressor")
    print(f"MSE (test): {mse:.4f}")

    # 5. Average MSE via 5-fold cross-validation
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    avg_mse = average_cv_mse(DecisionTreeRegressor(random_state=42), X, y, cv=5)
    print(f"Avg CV MSE: {avg_mse:.4f}")


if __name__ == "__main__":
    main()
