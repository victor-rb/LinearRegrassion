import pandas as pd
import joblib as jbl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

data = pd.read_csv('real_car_models.csv')

x = data[['model', 'year']]
y = data['price']

preprocessor = ColumnTransformer(
    transformers = [
        ('model', OneHotEncoder(), ['model'])
    ],
    remainder='passthrough'
)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

pipeline.fit(x_train, y_train)

y_pred = pipeline.predict(x_test)

mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")

jbl.dump(pipeline, 'linear_regression_model.pkl')