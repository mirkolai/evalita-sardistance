from gensim.models import Word2Vec
import os
from nltk.corpus import stopwords
import re

class ItalianTwitterWord2Vect(object):

    word2vec=None

    def create_model(self,file_name):
        file_name_output=file_name.split(".")[0]+'_Word2Vect.bin'
        if os.path.isfile(file_name_output):
            self.word2vec = Word2Vec.load(file_name_output)
        else:

            italianStopWords = {}  # faster indexing
            for stopword in stopwords.words('italian'):
                italianStopWords[stopword]=1

            i=0
            file=open(file_name)
            all_words=[]
            for text in file:
                i=i+1
                if i%10000==0:
                    print(i)

                # Cleaing the text
                text = text.lower()
                text = re.sub('[^0-9a-zàáâãäåçèéêëìíîïòóôõöùúûüă]', ' ', text )
                text = re.sub(r'\s+', ' ', text)

                # Preparing the dataset
                words = [ word for word in text.split(" ") if len(word)>1 and len(word)<50 and word not in italianStopWords]

                if len(words)<=50:
                    all_words.append(words)

            #creating model

            self.word2vec = Word2Vec(all_words, min_count=50)
            self.word2vec.save(file_name_output)

    def get_model(self):
         return  self.word2vec


if __name__ == '__main__':
    """
    tweets_2019.txt is the list of all tweets gathered using the Twitter's Stream API during the year 2019 (about 300M tweets).
    We used the vowels 'a','e','i','o','u' as keywords and the language filter 'it'.
    We discharge retweets, quotes, replies, and tweets containg urls or media.
    The resulting list includes about 16M tweets.
    We can not share the textual content of these tweets (https://developer.twitter.com/en/developer-terms/agreement-and-policy)
    For this reason, we release only the pre-trained word2vect model
    """
    file_name='model/tweets_2019.txt'
    italianTwitterWord2Vect=ItalianTwitterWord2Vect()
    italianTwitterWord2Vect.create_model(file_name)
    word2vec=italianTwitterWord2Vect.get_model()
    vocabulary = word2vec.wv.vocab
    print(word2vec.most_similar(positive=['sardine']))

