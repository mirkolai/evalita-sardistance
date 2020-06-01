import numpy
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from keras.layers import  SpatialDropout1D, LSTM, Dense, Embedding
from keras.callbacks import EarlyStopping
from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from neuralnetwork.ItalianTwitterWord2Vect import ItalianTwitterWord2Vect
from utils import Database_manager
from sklearn.metrics.classification import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import KFold
"""
You could use this script for evaluating your feature using the K-fold validation on training set
"""

# initialize database_manager
database_manager = Database_manager.make_database_manager()
# recover tweets
tweets = numpy.array(database_manager.return_tweets_training())
labels = numpy.array(database_manager.get_label(tweets))
encoder = LabelBinarizer()
labels = encoder.fit_transform(labels)
#load pretrained word2vect
italianTwitterWord2Vect = ItalianTwitterWord2Vect()
italianTwitterWord2Vect.create_model("neuralnetwork/model/tweets_2019.txt")
word2vec = italianTwitterWord2Vect.get_model()
NUM_WORDS=len(word2vec.wv.vocab)
#tokenize tweets in sequences of indexes
tokenizer = Tokenizer(num_words=NUM_WORDS,
                      lower=True)
tokenizer.fit_on_texts([tweet.text for tweet in tweets])
sequences = tokenizer.texts_to_sequences([tweet.text for tweet in tweets])
word_index = tokenizer.word_index
X = pad_sequences(sequences)
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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#k-fold validation
golden=[]
predict=[]
kf = KFold(n_splits=5, random_state=True)
for index_train, index_test in kf.split(tweets):

    sequence_length = X.shape[1]
    embedding_layer = Embedding(vocabulary_size,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                trainable=True)

    callback = EarlyStopping(monitor='val_loss', patience=3)  # for preventing overfitting

    model.fit(X[index_train], labels[index_train],
              batch_size=100, epochs=10,
              verbose=1, callbacks=[callback])

    test_predict = model.predict(X[index_test])


    current_golden = encoder.inverse_transform(labels[index_test])
    current_test_predict=encoder.inverse_transform(test_predict)

    golden=numpy.concatenate((golden,current_golden), axis=0)
    predict=numpy.concatenate((predict,current_test_predict), axis=0)


prec, recall, f, support = precision_recall_fscore_support(
golden,
predict,
beta=1)

accuracy = accuracy_score(
golden,
predict
)

print(prec)
print(recall)
print(f)
print("f[0]+f[1])/2: ",(f[0]+f[1])/2)
print(support)
print(accuracy)
