import pandas as pd
import sklearn

#data definitiion and filtering
data = pd.read_csv("dataset-inc-both-sexes-in-2022-all-cancers")

X = data.drop("GDP",axis=1)
y = data["Number"]
X = X.select_dtypes(include="number")

#insert train_test_split for training (non negotiable)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error

accuracy = mean_squared_error(y_test, y_pred)
print("Accuracy:", accuracy)

