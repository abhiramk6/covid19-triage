import pandas as pd
from MLP import test_data_accuracy, training_data_accuracy

corona_data = pd.read_csv('data/dataset2.csv')

print("dataset")
print(corona_data.head())
print("")

print("shape of data")
print(corona_data.shape)
print("")

print("data info")
print(corona_data.info())
print("")

print("training data accuracy:", training_data_accuracy)
print("testing data accuracy", test_data_accuracy)


