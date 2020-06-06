from machinelearning import Resource_user_info
from machinelearning import Resource_tweet_info
from machinelearning import Resource_quote_network
from machinelearning import Resource_reply_network
from machinelearning import Resource_retweet_network
from machinelearning import Resource_friend_network

users_info=Resource_user_info.User_Info()
tweets_info=Resource_tweet_info.Tweet_Info()
network_quote=Resource_quote_network.Quote_Network()
network_reply=Resource_reply_network.Reply_Network()
network_retweet=Resource_retweet_network.Retweet_Network()
network_friend=Resource_friend_network.Friend_Network()

class Tweet(object):

    id=None
    user_id=None
    text=None
    label=None
    user_info=None
    tweets_info=None
    community_quote=None
    community_reply=None
    community_retweet=None
    community_friend=None

    def __init__(self, id, user_id, text, label):
        self.id=id
        self.user_id=user_id
        self.text=text
        self.label=label
        self.user_info=users_info.get_user_info(user_id)
        self.tweet_info=tweets_info.get_tweet_info(id)
        self.community_quote=network_quote.get_network_community(user_id)
        self.community_reply=network_reply.get_network_community(user_id)
        self.community_retweet=network_retweet.get_network_community(user_id)
        self.community_friend=network_friend.get_network_community(user_id)

def make_tweet(id, user_id, text, label ):
    """
        Return a Tweet object.
    """
    tweet = Tweet(id, user_id, text, label)

    return tweet



