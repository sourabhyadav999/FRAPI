# -*- coding: utf-8 -*-
"""
Created on Sun May 10 11:54:26 2020

@author: sourabh
"""

from flask import Flask,jsonify,request
from flask_cors import CORS, cross_origin
from keras.layers import Dense , Flatten ,Embedding,Input,Conv1D,GlobalMaxPooling1D,Dropout
from keras.models import Sequential
import numpy as np
import re
import string
import ftfy
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


app = Flask(__name__)
cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'



printable = set(string.printable)
rep = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u',
       'v','w','x','y','z','A','B','B','D','E','F','G','H','I','J','K','L','M','N','O','P',
       'Q','R','S','T','U','V','W','X','Y','Z','!','@','#','$','%','&',',','?']
def clean_text(s):
    st=ftfy.fix_text(s)
    for s in rep:
        rs = s+s
        try:
            st = re.sub(rs+'+',rs,st)
        except:
            continue
            
    text = BeautifulSoup(st, 'lxml')
    text = text.get_text()
    text = text.lower()
    text = re.sub('["]',' ',text)
    text = re.sub('[|]',' ',text)
    text = re.sub('[-]',' ',text)
    text = re.sub('[.]',' ',text)
    text = re.sub("[']",' ',text)
    text = re.sub('https?://[A-Za-z0-9./]+',' ',text)
    text = re.sub('https ?: //[A-Za-z0-9./]+',' ',text)
    text = re.sub('http ?: //[A-Za-z0-9./]+',' ',text)
    text = re.sub('http?://[A-Za-z0-9./]+',' ',text)
    text = re.sub('pic.twitter.com[^ ]*',' ',text)
    text = re.sub('\.\.+','.',text)
    text = re.sub('[0-9]+',' ',text)
    text = re.sub(r'@[A-Za-z0-9]+',' ',text)
    text = re.sub(r'@ [A-Za-z0-9]+',' ',text)
    text = re.sub('@',' at ',text)
    text = re.sub('&',' and ',text)
    text = re.sub('w/',' with ',text)
    text = re.sub('[()]','',text)
    text = re.sub('#','',text)
    text = re.sub(':','',text)
    text = re.sub('_','',text)
    text = re.sub('[\\_]','',text)
    text = re.sub('\'re',' are',text)
    text = re.sub('i\'d','i would',text)
    text = re.sub('isn\'t','is not',text)
    text = re.sub('don\'t','do not',text)
    text = re.sub("i've",'i have',text)
    text = re.sub("it's",'it is',text)
    text = re.sub("wasn't",'was not',text)
    text = re.sub("can't",'can not',text)
    text = re.sub("haven't",'have not',text)
    text = re.sub("i'm",'i am',text)
    text = re.sub("we'll",'we will',text)
    text = re.sub("i'll",'i will',text)
    text = re.sub("let's",'let us',text)
    text = re.sub("imdrunk",'i am drunk',text)
    text = re.sub(' +',' ',text)
    return text







def create_model():
    model= Sequential()
    model.add(Embedding(112501 + 1,100,input_length=500))
    model.add(Conv1D(128,5,padding='valid',activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(30,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model












@app.route('/')
@cross_origin()
def home():
    return jsonify({ "prediction" : "online" })


@app.route('/prediction', methods=['POST'] )
@cross_origin()
def prediction():
    
    body = request.get_json()
    model=create_model()
    model.load_weights('frweights.h5')
    content = str(body['text'])
    text=clean_text(content)
    texts=[text]
    tokenizer = Tokenizer(num_words=None)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=500)
    ypred=model.predict(data)
    t=ypred[0][0]
    if t<0.5:
        exp="Real"
    else:
        exp="Fake"
   
    
    
    
    return jsonify({ "prediction" : exp })



if __name__=='__main__':
    app.run(debug=False)
