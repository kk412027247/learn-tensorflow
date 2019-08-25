from __future__ import print_function

import numpy as np
import pandas as pd

print(pd.__version__)

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 483199])
print(city_names, population)

california_housing_dataframe = pd.read_csv(
    'https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv', sep=',')
print(california_housing_dataframe.describe())
print(california_housing_dataframe.head())
california_housing_dataframe.hist('housing_median_age')

cities = pd.DataFrame({'City name': city_names, 'Population': population})
print(cities)

print(type(cities['City name']))
print(cities['City name'])

print(type(cities['City name'][1]))
print(cities['City name'][1])

print(type(cities[0:2]))
print(cities[0:2])

print(population / 1000)
print('---')
np.log(population)

print(population.apply(lambda val: val > 1000000))

cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
print(cities)

cities['Is wide and has saint name'] = (cities['Area square miles'] > 50) & cities['City name'].apply(
    lambda name: name.startswith('San'))
print(cities)

print(city_names.index)

print(cities.index)

print(cities.reindex([2, 0, 1]))

print(cities.reindex(np.random.permutation(cities.index)))

print(cities.reindex([3, 4, 1, 5]))
