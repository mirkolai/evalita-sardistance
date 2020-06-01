import numpy
import csv
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from keras.layers import  SpatialDropout1D, LSTM, Dense, Embedding
from keras.callbacks import EarlyStopping
from keras import Sequential
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from neuralnetwork.ItalianTwitterWord2Vect import ItalianTwitterWord2Vect
from utils import Database_manager
"""
You could use this script for predicting test set
"""

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



# initialize database_manager
database_manager = Database_manager.make_database_manager()
#recover tweets
tweets_train = numpy.array(database_manager.return_tweets_training())
labels_train = numpy.array(database_manager.get_label(tweets_train))
encoder = LabelBinarizer()
labels_train = encoder.fit_transform(labels_train)

tweets_test = numpy.array(database_manager.return_tweets_test())
#load pretrained word2vect
italianTwitterWord2Vect = ItalianTwitterWord2Vect()
italianTwitterWord2Vect.create_model("neuralnetwork/model/tweets_2019.txt")
word2vec = italianTwitterWord2Vect.get_model()
NUM_WORDS=len(word2vec.wv.vocab)
#tokenize tweets in sequences of indexes
tokenizer = Tokenizer(num_words=NUM_WORDS,
                      lower=True)
tokenizer.fit_on_texts([tweet.text for tweet in tweets_train])
sequences_train = tokenizer.texts_to_sequences([tweet.text for tweet in tweets_train])
word_index = tokenizer.word_index
X = pad_sequences(sequences_train)

sequences_test = tokenizer.texts_to_sequences([tweet.text for tweet in tweets_test])
y = pad_sequences(sequences_test)

#creation of the embedding matrix based on word2vect
for key in word2vec.wv.vocab:
    EMBEDDING_DIM=len(word2vec[key])
    break
vocabulary_size=min(len(word_index)+1,NUM_WORDS)
embedding_matrix = numpy.zeros((vocabulary_size, EMBEDDING_DIM))
for word, i in word_index.items():
    if i>=NUM_WORDS:
        continue
    try:
        embedding_vector = word2vec[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i]=numpy.random.normal(0,numpy.sqrt(0.25),EMBEDDING_DIM)
embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)
#model definition
model = Sequential()
model.add(embedding_layer)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[f1])

#training
sequence_length = X.shape[1]
embedding_layer = Embedding(vocabulary_size,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                trainable=True)

callback = EarlyStopping(monitor='val_loss', patience=3) #for preventing overfitting

model.fit(X, labels_train,
      batch_size=100, epochs=10,
      verbose=1, callbacks=[callback])

#prediction
test_predict = model.predict(y)


test_predict=encoder.inverse_transform(test_predict)

csvfile = open('neuralnetwork/results/TEST_PREDICTION.tsv', 'w', newline='')
spamwriter = csv.writer(csvfile, delimiter='\t',quotechar='"', quoting=csv.QUOTE_MINIMAL)
spamwriter.writerow(['tweet_id', 'label'])
for i in range(0,len(tweets_test)):
    spamwriter.writerow([tweets_test[i].id, test_predict[i]])


