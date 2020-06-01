from utils.Tweet import make_tweet
import csv

class Database_manager(object):

    def get_label(self,tweets):
        """
        Returns un array containing the label for each tweet in tweets
        :param tweets:  Array of Tweet Objects
        :return: Array of label
        """
        return [ tweet.label for tweet  in tweets]

    def return_tweets_training(self):
        """Returns an array containing a list of trainig tweets.
           Tweets are encoded as Tweet objects.
        """
        tweets=[]
        csvfile= open("data/TRAIN.csv")
        next(csvfile)#skip header
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')

        for tweet in spamreader:
                id=tweet[0]
                user_id=tweet[1]
                text=tweet[2]
                label=tweet[3]
                """
                Create a new istance of a Tweet object
                """
                this_tweet=make_tweet(id,user_id, text, label)
                tweets.append(this_tweet)
        return tweets


    def return_tweets_test(self):
        """Returns an array containing a list of testing tweets.
           No label is available.
           Tweets are encoded as Tweet objects.
        """
        tweets=[]
        csvfile= open("data/TEST.csv")
        next(csvfile)#skip header
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for tweet in spamreader:
                id=tweet[0]
                user_id=tweet[1]
                text=tweet[2]
                label=None
                """
                Create a new istance of a Tweet object
                """
                this_tweet=make_tweet(id,user_id, text, label)
                tweets.append(this_tweet)
        return tweets

    def return_tweets_test_labeled(self):
        """Returns an array containing a list of testing tweets.
           The label is available.
           Tweets are encoded as Tweet objects.
        """
        tweets=[]
        csvfile= open("data/TEST_labeled.csv")
        next(csvfile)#skip header
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for tweet in spamreader:
                id=tweet[0]
                user_id=tweet[1]
                text=tweet[2]
                label=tweet[3]
                """
                Create a new istance of a Tweet object
                """
                this_tweet=make_tweet(id,user_id, text, label)
                tweets.append(this_tweet)
        return tweets

def make_database_manager():
    database_manager = Database_manager()

    return database_manager




