import sys
import json
import numpy as np
import gensim
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input,Bidirectional,LSTM,TimeDistributed,Dense,Conv2D,MaxPooling2D,Flatten
from keras.optimizers import SGD,Adam
from keras_contrib.layers.crf import CRF
from keras.utils import to_categorical
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from keras.callbacks import ModelCheckpoint, EarlyStopping
label4node = [ "null" ,"value", "agent", "condition", "theme", "theme_mod", "quant_mod", "co_quant", "location", "whole", "source", "reference_time", "quant", "manner", "time", "cause"]	
label4edge = [ "equivalence" , "fact" , "analogy" ]

def getFeature(tokens,model): #tokens of data are stored as dictionary
    feature = np.zeros((1,len(tokens),300))
    # sentence to 2d feature vectors
    for s in range(len(tokens)):
        if tokens[s] in model.vocab:
            feature[0,s,:] =  model[tokens[s]]
        else:
            feature[0,s,:] =  model[random.choice(model.wv.index2entity)]
    return feature

def getAns(nodes,token_len):
    label = np.zeros((1,token_len,1))
    for n in nodes:
        scope = n[0]
        for indx in range(scope[0],scope[1]):
            for key in n[1]:
                label[0,indx,0] = label4node.index(key)
                break
    return label

def load_w2v():
    print('load word2vec model with GoogleNews as corpus...')
    model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

    return model

def json2dic(data):
	return json.loads(data)

def extract_token_node_edges(filename): # Extract tokens,nodes,edges information,stored by dic, from json file 
    print('read file and store the data with tokens,nodes,edges...')
    tokens , nodes , edges = [] , [] , []
    with open(filename) as f:
        data = f.readlines()
        for i in range(len(data)):
            d = json2dic(data[i])
            tokens.append(d["tokens"])
            nodes.append(d["nodes"])
            edges.append(d["edges"])
    return tokens,nodes,edges
    
def text_preprocessing(t):
    return t

def read_data(filename,w2v_model):
	tokens,nodes,edges = extract_token_node_edges(filename)
	tokens = text_preprocessing(tokens)
	return tokens,nodes,edges

def task1_data_generator(rawdata,model): 
     x , y = rawdata[0],rawdata[1]
     i = 0
     while(True):
         sentence , label = x[i] , y[i]
         Input , Label = np.zeros((1,len(sentence),300)),np.zeros((1,len(sentence),1))
         # sentence to 2d feature vectors
         for s in range(len(sentence)):
            if sentence[s] in model.vocab:
                Input[0,s,:] =  model[sentence[s]]
            else:
                Input[0,s,:] =  model[random.choice(model.wv.index2entity)]
         # node label to label vector 
         for n in label:
            scope = n[0]
            for indx in range(scope[0],scope[1]):
                for key in n[1]:
                    Label[0,indx,0] = label4node.index(key)
                    break
         i = i + 1
         if i == len(x):
             i=0
         yield Input, to_categorical(Label, num_classes= len(label4node))

def task2_data_generator_w2v(rawdata,w2v_model,lstm_model):
     x , y = rawdata[0],rawdata[1]
     Input , Label = [] , []
     for i in range(len(x)):
         sentence , edges = x[i] , y[i]
         for info in edges:
             node1,node2 = sentence[info[0][0]:info[0][1]],sentence[info[1][0]:info[1][1]]
             vec1,vec2 = np.zeros((300,)),np.zeros((300,))
             for w in node1:
                 if w in w2v_model.vocab:
                    vec1 = vec1 + w2v_model[w]
                 else:
                    vec1 = vec1 + w2v_model[random.choice(w2v_model.wv.index2entity)]
             for w in node2:
                if w in w2v_model.vocab:
                    vec2 = vec2 + w2v_model[w]
                else:
                    vec2 = vec2 + w2v_model[random.choice(w2v_model.wv.index2entity)]
             Input.append(np.concatenate((vec1, vec2)))
             for key in info[2]:
                 Label.append(label4edge.index(key))
                 break
     return np.array(Input),to_categorical(np.array(Label), num_classes= len(label4edge))

def task2_data_generator_w2lstm(rawdata,w2v_model,lstm_model):
     x , y = rawdata[0],rawdata[1]
     Input , Label = [] , []
     for i in range(len(x)):
         sentence , edges = x[i] , y[i]
         lstm_vector = np.zeros((1,len(sentence),300))
         # sentence to 2d feature vectors
         for s in range(len(sentence)):
            if sentence[s] in w2v_model.vocab:
                lstm_vector[0,s,:] =  w2v_model[sentence[s]]
            else:
                lstm_vector[0,s,:] =  w2v_model[random.choice(w2v_model.wv.index2entity)]
         feature_vec = lstm_model.predict(lstm_vector)
         for info in edges:
             vec1,vec2 = np.zeros((300,)),np.zeros((300,))
             for i in range(info[0][0],info[0][1]):
                 vec1 = vec1 + feature_vec[0,i,:]
             for i in range(info[1][0],info[1][1]):
                 vec2 = vec2 + feature_vec[0,i,:]
             Input.append(np.concatenate((vec1, vec2)))
             for key in info[2]:
                 Label.append(label4edge.index(key))
                 break
     return np.array(Input),to_categorical(np.array(Label), num_classes= len(label4edge))

def task1_build_model(d_dim,l_dim):
    print('task1 model building...')
    
    INPUT = Input(shape = (None,d_dim))
    BILSTM1 = Bidirectional(LSTM(300, return_sequences=True),merge_mode='sum',input_shape=(None,d_dim))(INPUT)
    BILSTM2 = Bidirectional(LSTM(300, return_sequences=True),merge_mode='sum',input_shape=(None,d_dim))(BILSTM1)
    TD = TimeDistributed(Dense(l_dim))(BILSTM2)
    FC1 = Dense(units = l_dim, activation = 'softmax')(TD)

    model = Model(INPUT,FC1)
    high_feature = Model(INPUT,BILSTM2)
    
    return model,high_feature

def task2_build_model(d_dim,l_dim):
    print('task2 model building...')
    
    INPUT = Input(shape = (d_dim,))
    FC1 = Dense(units = 300, activation = 'relu')(INPUT)
    FC2 = Dense(units = 300, activation = 'relu')(FC1)
    FC3 = Dense(units = l_dim, activation = 'softmax')(FC2)
    model = Model(INPUT,FC3)

    return model

if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 2:
        raise Exception('please enter train/test filename respectively')
    train_filename = argv[1]
    testing_filename = argv[2]
    
    w2v_model = load_w2v()
    tokens_train ,node_train ,edge_train = read_data(train_filename,w2v_model)
    tokens_test ,node_test ,edge_test  = read_data(testing_filename,w2v_model)
    
    ########################## Node label prediction given Tokens################################
    l_dim,d_dim = len(label4node), 300
    X_train, X_valid, y_train, y_vaild =train_test_split(tokens_train, node_train, test_size=0.33, random_state=42)
    Num_traindata ,Num_validdata = len(X_train),len(X_valid)
    g1 = task1_data_generator((X_train,y_train),w2v_model)
    g2 = task1_data_generator((X_valid,y_vaild),w2v_model)

    model4node , w2lstm_model = task1_build_model(d_dim,l_dim)
    model4node.summary()
    model4node.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    with open('model4node.json','w') as f:    # save the model
        f.write(model4node.to_json()) 
    ckpt = ModelCheckpoint('model4node',monitor='val_acc',save_best_only=True,save_weights_only=True,verbose=1)
    cb= [ckpt]    
    model4node.fit_generator(g1,steps_per_epoch=Num_traindata,
                                        epochs=5,
                                        verbose=1,
                            validation_data=g2,
                            max_queue_size = 15,
                            validation_steps=Num_validdata,callbacks=cb,
                            )	
    
    ########################## edge label prediction given nodes################################
    l_dim, d_dim = len(label4edge), 600
    X_train, X_valid, y_train, y_vaild =train_test_split(tokens_train, edge_train, test_size=0.33, random_state=42)
    Num_traindata ,Num_validdata = len(X_train),len(X_valid)
    X_train,y_train = task2_data_generator_w2lstm((X_train,y_train),w2v_model,w2lstm_model)
    X_valid,y_vaild = task2_data_generator_w2lstm((X_valid,y_vaild),w2v_model,w2lstm_model)
    
    model4edge = task2_build_model(d_dim,l_dim)
    model4edge.summary()
    model4edge.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    with open('model4edge.json','w') as f:    # save the model
        f.write(model4edge.to_json()) 
    ckpt = ModelCheckpoint('model4edge',monitor='val_acc',save_best_only=True,save_weights_only=True,verbose=1)
    cb= [ckpt]    
    model4edge.fit(X_train,y_train,validation_data=(X_valid,y_vaild),callbacks=cb,batch_size=100,epochs=20)
    
    ########################## make prediction on testing data################################
    y_true , y_pred = [] , []
    for i in range(len(tokens_test)):
        feature_vec = getFeature(tokens_test[i],w2v_model)
        ans = getAns(node_test[i],len(tokens_test[i]))
        ans = np.reshape(ans, (ans.shape[1],)).astype('int')
        y_true = y_true + ans.tolist()
        
        pred = model4node.predict(feature_vec)
        pred = np.reshape(pred, (pred.shape[1], pred.shape[2]))
        pred = np.argmax(pred, axis=1)
        y_pred = y_pred + pred.tolist()
    evaluation = precision_recall_fscore_support(y_true, y_pred, average='macro')
    print("evaluation on node label prediction...")
    print(evaluation)        
    
    X_test,y_true = task2_data_generator_w2lstm((tokens_test,edge_test),w2v_model,w2lstm_model)
    y_true = np.argmax(y_true, axis=1)
    y_pred = model4edge.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    evaluation = precision_recall_fscore_support(y_true, y_pred, average='macro')
    print("evaluation on node label prediction...")    
    print(evaluation)
    

    
    
    
    
    
    
    
        