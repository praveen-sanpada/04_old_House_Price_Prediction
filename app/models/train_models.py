# train_models.py
from app.utils.data_preprocessing import preprocess_data
from app.utils.model_training import (
    train_linear_regression, train_polynomial_regression, save_model
)

X_train, X_test, y_train, y_test, scaler = preprocess_data('data/house_data.csv')

lr_model = train_linear_regression(X_train, y_train)
save_model(lr_model, 'app/models/linear_regression_model.pkl')

poly_model = train_polynomial_regression(X_train, y_train, degree=2)
save_model(poly_model, 'app/models/polynomial_regression_model.pkl')
