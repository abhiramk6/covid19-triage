import numpy as np
from logistic_regression_model import model


print("")
print("enter below data in integers")
input_1 = input("enter your age: ")
input_2 = int(input("do you have any chronic disease (enter 1 for YES and 0 for NO): "))
input_3 = int(input("do you have any respiratory problems (enter 1 for YES and 0 for NO): "))
input_4 = int(input("do you have any cardiac problems (enter 1 for YES and 0 for NO): "))
input_5 = int(input("do you have any kidney problems (enter 1 for YES and 0 for NO): "))
input_6 = int(input("do you have any diabetic history (enter 1 for YES and 0 for NO): "))


input_data = (input_1, input_2, input_3, input_4, input_5, input_6)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)



prediction = model.predict(input_data_reshaped)

print("")

if prediction[0] == 1:
    print('The Person should be admitted in ICU')

elif prediction[0] == 0 and input_data[1] == 1:
    print('The Person should be admitted in Motor ventilation')

elif prediction[0] == 0 and input_data[1] == 0:
    print('The Person should be admitted in Normal Ward')

print("")
