from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


home_data = pd.read_csv("train.csv")

y = home_data.SalePrice
features = ['GrLivArea','LotArea', 'OverallQual', 'OverallCond', 'HeatingQC', 'CentralAir', ]
X = home_data[features]

#mapping data's strings to integers
heatMapping = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po':0}
acMapping = {'Y': 1, 'N': 0}
X = X.replace({'HeatingQC': heatMapping, 'CentralAir' : acMapping})

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
l_model = LinearRegression().fit(train_X, train_y)
rf_model = RandomForestRegressor(random_state = 1).fit(train_X, train_y)
#print("Model weights: ")
#print(l_model.coef_)

l_predictions_val = l_model.predict(val_X)
l_predictions_train = l_model.predict(train_X)

rf_predictions_val = rf_model.predict(val_X)
rf_maeVal = mean_absolute_error(rf_predictions_val, val_y)

l_maeVal = mean_absolute_error(l_predictions_val, val_y)
l_maeTrain = mean_absolute_error(l_predictions_train, train_y)

print("Linear Regression Validation mae: " + str(l_maeVal))
print("Random Forrest Validations mae: " + str(rf_maeVal))
print("Linear Regression Training mae: " + str(l_maeTrain))


#running on test data
test_data = pd.read_csv("test.csv")
test_X = test_data[features]

heatMapping = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po':0}
acMapping = {'Y': 1, 'N': 0}
test_X = test_X.replace({'HeatingQC': heatMapping, 'CentralAir' : acMapping})

#test_X['Utilities'] = test_X['Utilities'].fillna(0)
#print(test_X.Utilities.isnull().any())
#print(test_X.loc[[484]])
#test_X = test_X.fillna(test_X.mean)

test_preds = rf_model.predict(test_X)
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice' : test_preds})
output.to_csv("submission.csv", index= False)
