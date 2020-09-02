from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

home_data = pd.read_csv("train.csv")

y = home_data.SalePrice
features = ['LotArea', 'Utilities', 'OverallQual', 'OverallCond', 'HeatingQC', 'CentralAir']
X = home_data[features]

#mapping data's strings to integers
utilMapping = {'AllPub' : 3, 'NoSewr': 2, 'NoSeWa': 1, 'ELO': 0}
heatMapping = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po':0}
acMapping = {'Y': 1, 'N': 0}
X = X.replace({'Utilities': utilMapping, 'HeatingQC': heatMapping, 'CentralAir' : acMapping})

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
model = LinearRegression().fit(train_X, train_y)

print("Model weights: ")
print(model.coef_)
predictions_val = model.predict(val_X)
predictions_train = model.predict(train_X)

maeVal = mean_absolute_error(predictions_val, val_y)
maeTrain = mean_absolute_error(predictions_train, train_y)
print("Validation mae: " + str(maeVal))
print("Training mae: " + str(maeTrain))

#running on test data
test_data = pd.read_csv("test.csv")
test_X = test_data[features]

utilMapping = {'AllPub' : 3, 'NoSewr': 2, 'NoSeWa': 1, 'ELO':0}
heatMapping = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po':0}
acMapping = {'Y': 1, 'N': 0}
test_X = test_X.replace({'Utilities': utilMapping, 'HeatingQC': heatMapping, 'CentralAir' : acMapping})

test_X['Utilities'] = test_X['Utilities'].fillna(0)
#print(test_X.Utilities.isnull().any())
#print(test_X.loc[[484]])
#test_X = test_X.fillna(test_X.mean)

test_preds = model.predict(test_X)
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice' : test_preds})
output.to_csv("submission.csv", index= False)