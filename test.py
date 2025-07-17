import mlflow

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

mlflow.autolog()
data_diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data_diabetes.data, data_diabetes.target)

# Create and train models.
reg = RandomForestRegressor(n_estimators=120, max_depth=10, max_features=5)
reg.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = reg.predict(X_test)
