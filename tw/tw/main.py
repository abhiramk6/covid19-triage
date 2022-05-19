import numpy as np

from django.http import HttpResponse
from django.shortcuts import render



def mlp():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score



    corona_data = pd.read_csv('/Users/abhiramkamini/python_files/django_all/triageweb/tw/tw/data/dataset2.csv')

    X = corona_data.drop(['death_binary', 'ID'], axis=1)
    Y = corona_data['death_binary']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    model = MLPClassifier(hidden_layer_sizes=(15,), random_state=1, max_iter=2, warm_start=True)
    for i in range(450):
         model.fit(X, Y)



    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)



   # print(training_data_accuracy,test_data_accuracy)


    return model


model=mlp()





arr=[]





def index(request):
    return render(request,"index.html")




def condition(request):
    arr.append( request.POST.get('a','0'))
    arr.append( request.POST.get('b', 'off'))
    arr.append( request.POST.get('c', 'off'))
    arr.append( request.POST.get('d', 'off'))
    arr.append( request.POST.get('e', 'off'))
    arr.append( request.POST.get('f', 'off'))
    arr.append( request.POST.get('g', 'off'))
    arr.append( request.POST.get('i', 'off'))
    arr.append( request.POST.get('j', 'off'))
    arr.append( request.POST.get('k', 'off'))
    arr.append( request.POST.get('l', 'off'))
    arr.append( request.POST.get('m', 'off'))
    arr.append( request.POST.get('n', 'off'))
    arr.append( request.POST.get('o', 'off'))
    arr.append( request.POST.get('p', 'off'))
    arr.append( request.POST.get('q', 'off'))
    arr.append( request.POST.get('r', 'off'))
    arr.append( request.POST.get('s', 'off'))
    arr.append( request.POST.get('t', 'off'))
    arr.append( request.POST.get('u', 'off'))
    arr.append( request.POST.get('v', 'off'))
    arr.append( request.POST.get('w', 'off'))
    arr.append( request.POST.get('x', 'off'))





    for i in range(1,23):
        if arr[i]=="on":
            arr[i]=1

        elif arr[i]=='off':
            arr[i]=0

    arr[0] = int(arr[0])



    input_data = ( arr[0], arr[1],arr[2],arr[3], arr[4],arr[5],arr[6],arr[7],arr[8],arr[9],arr[10],arr[11],arr[12],arr[13],arr[14],arr[15],arr[16],arr[17],arr[18],arr[19],arr[20],arr[21],arr[22])
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = model.predict(input_data_reshaped)


    print(input_data)


    if prediction[0] == 1:
        par=('The Person should be admitted in ICU')
        print(par,prediction[0])

    if prediction[0] == 0 and input_data[1] == 1:
        par=('The Person should be admitted in Motor ventilation')
        print(par,prediction[0],input_data[1])

    elif prediction[0] == 0 and input_data[1] == 0:
        par=('The Person should be admitted in Normal Ward')
        print(par,prediction[0],input_data[1],arr[1])


    params={'param':par}

    input_data=()
    arr.clear()

    return render(request, 'condition.html', params)



















def bruteforce():

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
