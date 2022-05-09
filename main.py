import numpy as np
from MLP import model



input_data=(45,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1)
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
