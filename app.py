# Importing necessary libraries
from flask import Flask, render_template, request, url_for, flash, redirect
import pandas as pd 
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/drug')
def drug():
    return render_template('drug.html')

@app.route('/load',methods=["GET","POST"])
def load():
    global df, dataset
    if request.method == "POST":
        data = request.files['data']
        df = pd.read_csv(data)
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load.html', msg=msg)
    return render_template('load.html')


@app.route('/view')
def view():
    print(dataset)
    print(dataset.head(2))
    print(dataset.columns)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())


@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  hvectorizer,df
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        
       # Assigning the value of x and y 
        x = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.3, random_state=72)

        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)

    
        print(x_train,x_test)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')

@app.route('/model', methods=['POST', 'GET'])
def model():
    if request.method == "POST":
        global model
        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s = int(request.form['algo'])
        if s == 0:
            return render_template('model.html', msg='Please Choose an Algorithm to Train')
        elif s == 1:
            print('aaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            clf = LogisticRegression()
            clf.fit(x_train,y_train)
            model = clf
            y_pred = clf.predict(x_test)
            ac_lr = accuracy_score(y_test, y_pred)
            ac_lr = ac_lr * 100
            print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            msg = 'The accuracy obtained by Logistic Regression is  ' + str(ac_lr) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 2:
            classifier = DecisionTreeClassifier(max_leaf_nodes=39)
            classifier.fit(x_train, y_train)
            y_pred  =  classifier.predict(x_test)            
            
            ac_dt = accuracy_score(y_test, y_pred)
            ac_dt = ac_dt * 100
            msg = 'The accuracy obtained by Decision Tree Classifier is ' + str(ac_dt) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 3:
            classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5,
                                       n_estimators=100, oob_score=True)
            classifier_rf.fit(x_train, y_train)
            y_pred  =  classifier_rf.predict(x_test)            
            
            ac_rf = accuracy_score(y_test, y_pred)
            ac_rf = ac_rf * 100
            msg = 'The accuracy obtained by Random Forest Classifier is ' + str(ac_rf) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 4:
            clf_et = ExtraTreesClassifier()
            clf_et.fit(x_train, y_train)
            y_pred  =  clf_et.predict(x_test)            
            
            ac_et = accuracy_score(y_test, y_pred)
            ac_et = ac_et * 100
            msg = 'The accuracy obtained by Extra Tree Classifier is ' + str(ac_et) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 5:
            xgb = XGBClassifier()
            xgb.fit(x_train, y_train)
            y_pred  =  xgb.predict(x_test)            
            
            ac_xgb = accuracy_score(y_test, y_pred)
            ac_xgb = ac_xgb * 100
            msg = 'The accuracy obtained by XGBoost Classifier is ' + str(ac_xgb) + str('%')
            return render_template('model.html', msg=msg)
    return render_template('model.html')

@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        # f1=int(request.form['city'])
        f1 = int(request.form['Category'])
        f2 = float(request.form['Size'])
        f3 = int(request.form['Type'])
        f4 = float(request.form['Price'])
        f5 = int(request.form['Content Rating'])
        f6 = int(request.form['Genres'])
        print(f2)
        print(type(f2))

        li = [f1,f2,f3,f4,f5,f6]
        print(li)
        
        # model.fit(X_transformed, y_train)
        
        # print(f2)
        import pickle
        model  = DecisionTreeClassifier(max_leaf_nodes=39)
        model.fit(x_train,y_train)
        result = model.predict([li])
        print(result)
        print('result is ',result)
        # (Popular  = 1,   Not popular  = 0 )
        if result == 0:
            msg = 'This is a Popular App'
            return render_template('prediction.html', msg=msg)
        else:
            msg = 'This is Not a Popular App'
            return render_template('prediction.html', msg=msg)
    return render_template('prediction.html')










if __name__=='__main__':
    app.run(debug=True)