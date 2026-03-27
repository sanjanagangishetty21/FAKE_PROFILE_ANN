from django.shortcuts import render
from django.conf import settings
import os
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam

model = None


def index(request):
    return render(request, 'index.html', {})

def User(request):
    return render(request, 'User.html', {})

def Admin(request):
    return render(request, 'Admin.html', {})

def AdminLogin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if username == 'admin' and password == 'admin':
            context = {'data': 'welcome ' + username}
            return render(request, 'AdminScreen.html', context)
        else:
            context = {'data': 'login failed'}
            return render(request, 'Admin.html', context)
        

def importdata():
    dataset_path = os.path.join(settings.BASE_DIR,
                                'Profile',
                                'dataset',
                                'dataset.txt')

    data = pd.read_csv(dataset_path)
    data = data.abs()
    return data


def splitdataset(data):
    X = data.values[:, 0:8]
    y_ = data.values[:, 8]

    y_ = y_.reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(y_)

    train_x, test_x, train_y, test_y = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    return train_x, test_x, train_y, test_y


def GenerateModel(request):
    global model

    data = importdata()
    train_x, test_x, train_y, test_y = splitdataset(data)

    model = Sequential()
    model.add(Dense(200, input_shape=(8,), activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_x, train_y,
              epochs=200,
              batch_size=5,
              verbose=2)

    results = model.evaluate(test_x, test_y)
    ann_acc = results[1] * 100

    model_path = os.path.join(settings.BASE_DIR, 'ann_model.h5')
    model.save(model_path)

    context = {'data': 'ANN Accuracy : ' + str(ann_acc)}
    return render(request, 'AdminScreen.html', context)


def UserCheck(request):
    global model

    if request.method == 'POST':

        if model is None:
            model_path = os.path.join(settings.BASE_DIR, 'ann_model.h5')
            model = load_model(model_path)

        user_input = request.POST.get('t1')

        values = list(map(float, user_input.split(',')))
        test = np.array([values])

        prediction = model.predict(test)
        predict = np.argmax(prediction, axis=1)

        if predict[0] == 0:
            msg = "Given Account Details Predicted As Genuine"
        else:
            msg = "Given Account Details Predicted As Fake"

        context = {'data': msg}
        return render(request, 'User.html', context)


def ViewTrain(request):
    dataset_path = os.path.join(settings.BASE_DIR,
                                'Profile',
                                'dataset',
                                'dataset.txt')

    data = pd.read_csv(dataset_path)

    strdata = '<table border=1 align=center width=100%><tr>'
    headers = ['Account Age', 'Gender', 'User Age',
               'Link Description', 'Status Count',
               'Friend Count', 'Location',
               'Location IP', 'Profile Status']

    for h in headers:
        strdata += f'<th><font size=4 color=white>{h}</font></th>'
    strdata += '</tr><tr>'

    rows, cols = data.shape

    for i in range(rows):
        for j in range(cols):
            strdata += '<td><font size=3 color=white>' + \
                       str(data.iloc[i, j]) + \
                       '</font></td>'
        strdata += '</tr><tr>'

    context = {'data': strdata}
    return render(request, 'ViewData.html', context)
