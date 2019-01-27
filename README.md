# IRIE_project_2018_relation_extraction
Relation Extraction with The Wall Street Journal

## Prerequisite
Gensim
Keras
sklearn

## workflow
* Transfer documents into feature vectors by 3-billion words Google News pre-trained word2vec model
* Train a Bidirectional LSTM following DNN model to predict Node Role/Label
* Use the output of Bidirectional LSTM layer as input to train a DNN model for edge label prediction

## model summary
### node role prediction model
![alt text](https://github.com/leduoyang/IRIE_project_2018_relation_extraction/blob/master/result/model_node.png)
### edge relation prediction model
![alt text](https://github.com/leduoyang/IRIE_project_2018_relation_extraction/blob/master/result/model_edge.png)

## result
![alt text](https://github.com/leduoyang/IRIE_project_2018_relation_extraction/blob/master/result/eval.png)


source:
https://arxiv.org/pdf/1809.02700.pdf

