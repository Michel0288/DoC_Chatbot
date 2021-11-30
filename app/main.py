#import modules 
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
from tensorflow.python.framework import ops
import random
import json
import pickle



#read data from file 
with open("app/datafile2.json") as file:
    data = json.load(file)
try:
    #prevent model from retraining if file exists
    with open("datafile.pickle", "rb") as f:
        words_list, tag_label, training, output = pickle.load(f)
except:
    #create lists to store data from file in lists
    words_list = []
    tag_label = []
    document_x = []
    document_y = []
    #ignore question mark and exclamations from statement/questions
    ignore = ['?', '!']

    #extract data we want from datafile and turn each pattern into a list of words using tokenizer rather than having them as strings 
    #then add pattern data to list
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words_list.extend(wrds)
            document_x.append(wrds)
            document_y.append(intent["tag"])

        if intent["tag"] not in tag_label:
            tag_label.append(intent["tag"])

    #get root of word ex. what's ==what
    words_list = [stemmer.stem(w.lower()) for w in words_list if w not in ignore]
    words_list = sorted(list(set(words_list)))
    tag_label = sorted(tag_label)
    training = []
    output = []
    out_empty = [0 for _ in range(len(tag_label))]

    #neural network doesnt use strings as inputs so words are converted to 0 and 1s
    for x, doc in enumerate(document_x):
        bag_words = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words_list:
            if w in wrds:
                bag_words.append(1)
            else:
                bag_words.append(0)

        output_row = out_empty[:]
        output_row[tag_label.index(document_y[x])] = 1

        training.append(bag_words)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)
    
    #prevent model from retraining if file exists
    with open("datafile.pickle", "wb") as f:
        pickle.dump((words_list, tag_label, training, output), f)

#
#create neural network 
ops.reset_default_graph()

net_model= tflearn.input_data(shape=[None, len(training[0])])
net_model= tflearn.fully_connected(net_model, 8)
net_model= tflearn.fully_connected(net_model, 8)
net_model= tflearn.fully_connected(net_model, len(output[0]), activation="softmax")
net_model= tflearn.regression(net_model)

model = tflearn.DNN(net_model)

#prevent model from retraining if file exists
try:
    model.load("modeldata.h5")
except:
    #train model 
    model.fit(training, output, n_epoch=200, batch_size=5, show_metric=True)
    model.save("modeldata.h5")
    #score  =  model.evaluate(training,  output)
    # print(model.evaluate(training,  output))

#make predictions from model
def prediction(user_input, words_list):
    bag_words = [0 for _ in range(len(words_list))]
    #converts input to group of wrds since the model takes a group of words and not a string
    s_words = nltk.word_tokenize(user_input)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for i in s_words:
        for x, w in enumerate(words_list):
            if w == i:
                bag_words[x] = 1
    return numpy.array(bag_words)

#chat with bot func 
def chat_bot(var_input):
    while True:
        results = model.predict([prediction(var_input, words_list)])
        results_index = numpy.argmax(results)
        tag = tag_label[results_index]
        #picks a suitable response from dataset class
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        return(random.choice(responses))
    
