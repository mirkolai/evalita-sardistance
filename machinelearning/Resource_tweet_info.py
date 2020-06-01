__author__ = 'mirko'
import csv

class Tweet_Info(object):
    tweet={}
    def __init__(self):
            """Return an array containing users.
               users are encoded as objects.
            """
            self.tweets = {}
            csvfile = open("data/TWEET.csv")
            next(csvfile)  # skip header
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for tweet in spamreader:
                #"tweet_id", "user_id", "retweet_count", "favorite_count", "created_at"

                self.tweets[tweet[0]] = {}
                self.tweets[tweet[0]]['retweet_count'] = int(tweet[2])
                self.tweets[tweet[0]]['favorite_count'] = int(tweet[3])
                self.tweets[tweet[0]]['source'] = tweet[4]
                self.tweets[tweet[0]]['created_at'] = tweet[5]

    def get_tweet_info(self,tweet_id):
        return self.tweets[tweet_id]

def make_tweet_info():
    tweet_info = Tweet_Info()

    return tweet_info
