import pandas as pd
import joblib as jbl
import numpy as np

new_data = pd.read_csv('new_real_car_models.csv')

pipeline = jbl.load('linear_regression_model.pkl')

prediction = pipeline.predict(new_data)
prediction = np.round(prediction).astype(int)

new_data['predicted_price'] = prediction

new_data.to_csv('predicted_car_prices.csv', index=False)

print(new_data[['model', 'year', 'predicted_price']])