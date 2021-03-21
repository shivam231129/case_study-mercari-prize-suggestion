#import libraries
import numpy as np
import pandas as pd
import re

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso
#from xgboost import XGBRegressor

from nltk.tokenize import word_tokenize
#import gensim.models

from scipy.sparse import hstack
from sklearn.metrics import mean_squared_log_error

from prettytable import PrettyTable

import joblib
import gc

import warnings
warnings.filterwarnings('ignore')
import pickle

from keras.utils import to_categorical

import tensorflow as tf

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense,Dropout,Embedding,LSTM
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.optimizers import Adam
from keras.models import Sequential
from tqdm import tqdm
import warnings
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from keras.initializers import he_normal
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import LSTM,Bidirectional
from keras.layers.core import Dense, Dropout
from keras.models import Model, load_model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau

import keras
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
import math
from keras.callbacks import *
from keras.models import load_model
import time
import flask
from flask import Flask, jsonify, request


 # Ref: AAIC Notebook for Donors' Choose
def decontracted(sent):
    
      '''
      Task:   This Function changes common short forms like can't, won't to can not, will not resp. (Decontraction)
              This is done to ensure uniformity in the whole text
      Input:  Raw Text
      Output: Decontracted Text
      '''
      sent = re.sub(r"aren\'t", "are not", sent)
      sent = re.sub(r"didn\'t", "did not", sent)
      sent = re.sub(r"can\'t", "can not", sent)
      sent = re.sub(r"couldn\'t", "could not", sent)
      sent = re.sub(r"won\'t", "would not", sent)
      sent = re.sub(r"wouldn\'t", "would not", sent)
      sent = re.sub(r"haven\'t", "have not", sent)
      sent = re.sub(r"shouldn\'t", "should not", sent)
      sent = re.sub(r"doesn\'t", "does not", sent)
      sent = re.sub(r"don\'t", "do not", sent)
      sent = re.sub(r"didn\'t", "did not", sent)
      sent = re.sub(r"mustn\'t", "must not", sent)
      sent = re.sub(r"needn\'t", "need not", sent)
      
      return sent



app = Flask(__name__)
  # load model

#Initializing global variables
configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)

number = 0
name = []
item_condition_id = []
category_name = []
brand_name = []
shipping = []
item_description = []

model =  load_model('model7.h5')
vectorizer3=joblib.load('vector3.pkl')
vectorizer4=joblib.load('vector4.pkl')
t_1=joblib.load('embname.pkl')
t_2=joblib.load('embtext.pkl')
columns = ['name', 'item_condition_id', 'brand_name', 'category_name', 'shipping', 'item_description']


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/index1', methods=['POST'])
def index1():
	global number 
	number = int(request.form['nm'])
	return flask.render_template('temp1/index.html')

    

@app.route('/predict',methods = ['POST'])
def predict():





  

    global number
    global name
    global item_condition_id
    global category_name
    global brand_name
    global shipping
    global item_description
    to_predict_list = request.form.to_dict()
    
    name.append(to_predict_list['name'])
    item_condition_id.append(int(to_predict_list['item_condition_id']))
    shipping.append(int(to_predict_list['shipping']))
    if to_predict_list['category_name']=="":
        category_name.append(math.nan)
    else:
        category_name.append(to_predict_list['category_name'])
    if to_predict_list['brand_name']=="":
        brand_name.append(math.nan)
    else:
        brand_name.append(to_predict_list['brand_name'])
    if to_predict_list['item_description']=="":
        item_description.append(math.nan)
    else:
        item_description.append(to_predict_list['item_description'])
    number = number-1
    if number!=0:
        return flask.render_template('temp1/index.html')
    else:

        start_time = time.time()
        X_test =  pd.DataFrame({'name' : name})
        X_test['item_condition_id'] = item_condition_id
        X_test['category_name'] = category_name
        X_test['shipping'] = shipping
        X_test['brand_name'] = brand_name
        X_test['item_description'] = item_description
	
        X_test['brand_name'] = X_test['brand_name'].fillna("Unknown Brand.")
        X_test["item_description"] = X_test["item_description"].fillna(value="No description yet.")
   
        X_test.fillna('', inplace=True)

        X_test['item_description']  = X_test['item_description'].str.replace('^no description yet$', '', regex=True)
  		#del all_text_no_punc
        X_test['category_name'] = X_test['category_name'].fillna(value="Unknown Category.")

        X_test['item_description']  = X_test['item_description'].str.replace('^no description yet$', '', regex=True)

        X_test['name'] = X_test['name'] + " " + X_test['brand_name']
        X_test['text'] = X_test['item_description'] + " " + X_test['name'] + " " + X_test['category_name']

        X_test['name'] = X_test['name'].apply(lambda x : decontracted(x))
        X_test['text'] = X_test['text'].apply(lambda x : decontracted(x))



    
        valid_shipvec = vectorizer3.transform(X_test['shipping'].values.reshape(-1, 1))

        valid_conditionvec = vectorizer4.transform(X_test['item_condition_id'].values.reshape(-1, 1))

        X_te = hstack((valid_shipvec, valid_conditionvec)).todense()
  
 # print("################################################# categorical feature encoding done #####################################################")




      #again tolenizing 
        max_length=300


        encoded_test_desc = t_2.texts_to_sequences(X_test['text'])
        padded_test_desc = pad_sequences(encoded_test_desc, maxlen=max_length, padding='post')
        encoded_test=t_1.texts_to_sequences(X_test['name'])
        padded_test=pad_sequences(encoded_test, maxlen=max_length, padding='post')


  #print("################################################# name and text encoding done #####################################################")


        test_2 = [padded_test,padded_test_desc,X_te]



  #print("^^^^ prediction^^^^^^")
        y_pred=model.predict(test_2)

 # print("predicted_value is =",y_pred)

  #end_time=time.time() 

  #print("time taken in seconds=",end_time-start_time)



        y_pred=np.expm1(y_pred)
        y_pred= y_pred[0].tolist() 


    #output = round(prediction[0], 2)
        return jsonify({'prediction_ $':y_pred })

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    data = np.array(data)

    X_test = pd.DataFrame([data], columns=columns)








  

    X_test.fillna('', inplace=True)

    X_test['item_description']  = X_test['item_description'].str.replace('^no description yet$', '', regex=True)

    X_test['name'] = X_test['name'] + " " + X_test['brand_name']
    X_test['text'] = X_test['item_description'] + " " + X_test['name'] + " " + X_test['category_name']

    X_test['name'] = X_test['name'].apply(lambda x : decontracted(x))
    X_test['text'] = X_test['text'].apply(lambda x : decontracted(x))



    
    valid_shipvec = vectorizer3.transform(X_test['shipping'].values.reshape(-1, 1))

    valid_conditionvec = vectorizer4.transform(X_test['item_condition_id'].values.reshape(-1, 1))

    X_te = hstack((valid_shipvec, valid_conditionvec)).todense()
  
 # print("################################################# categorical feature encoding done #####################################################")




      #again tolenizing 
    max_length=300
  #https://subscription.packtpub.com/book/application_development/9781782167853/1/ch01lvl1sec10/tokenizing-sentences-into-words
  #global t_1
  #t_1 = joblib.load('/content/drive/My Drive/new/embname.pkl')
  #t_1.fit_on_texts(name_train)
  #https://subscription.packtpub.com/book/application_development/9781782167853/1/ch01lvl1sec10/tokenizing-sentences-into-words
  #global t_2
  
  #t_2.fit_on_texts(text_train)
    encoded_test_desc = t_2.texts_to_sequences(X_test['text'])
    padded_test_desc = pad_sequences(encoded_test_desc, maxlen=max_length, padding='post')
    encoded_test=t_1.texts_to_sequences(X_test['name'])
    padded_test=pad_sequences(encoded_test, maxlen=max_length, padding='post')


  #print("################################################# name and text encoding done #####################################################")


    test_2 = [padded_test,padded_test_desc,X_te]

 # print("%%%%%%%%% load model%%%%%%")
  # load model
 # model =  load_model('/content/drive/My Drive/new/model7.h5')
  # summarize model.
  #model.summary()

  #print("^^^^ prediction^^^^^^")
    y_pred=model.predict(test_2)

 # print("predicted_value is =",y_pred)

  #end_time=time.time() 

  #print("time taken in seconds=",end_time-start_time)



    y_pred=np.expm1(y_pred)
    return jsonify(y_pred[0])

if __name__ == '__main__':
    app.run(debug=True)
