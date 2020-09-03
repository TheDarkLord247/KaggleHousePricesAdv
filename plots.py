from matplotlib import pyplot as plt
import pandas as pd

home_data = pd.read_csv("train.csv")

y = home_data.SalePrice
features = ['LotArea', 'OverallQual', 'OverallCond', 'HeatingQC', 'CentralAir', ]
X = home_data[features]

#mapping data's strings to integers
utilMapping = {'AllPub' : 3, 'NoSewr': 2, 'NoSeWa': 1, 'ELO': 0}
heatMapping = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po':0}
acMapping = {'Y': 1, 'N': 0}
X = X.replace({'HeatingQC': heatMapping, 'CentralAir' : acMapping})

home_data['BsmtCond'] = home_data['BsmtCond'].fillna('none')
plt.scatter(home_data.BsmtCond, y)
plt.show()

#checking for linearity