from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
from datetime import datetime
from scipy.sparse import csr_matrix, hstack

class Features_manager(object):

    def __init__(self):
        """You could add new feature types in global_feature_types_list
            global_feature_types_list is  a dictionary containing the feature space matrix for each feature type

            if you want to add a new feature:
            1. chose a keyword for defining the feature type
            2. define a function function_name(self,tweets,tweet_test=None) where:

                tweets: Array of  tweet objects belonging to the training set
                tweet_test: Optional. Array of tweet objects belonging to the test set

                return:
                X_train: The feature space of the training set (numpy.array)
                X_test: The feature space of the test set, if test  set was defined (numpy.array)
                feature_names: An array containing the names of the features used for creating the feature space (array)

        """
        self.global_feature_types_list={
            "unigram":          self.get_unigram_features,
            "n-gram_1-3":          self.get_ngram_1_3_features,
            "unigramhashtag" :    self.get_ngramshashtag_features,
            "chargrams":         self.get_nchargrams_features,
            "numhashtag":        self.get_numhashtag_features,
            "puntuactionmarks":  self.get_puntuaction_marks_features,
            "length":            self.get_length_features,
            "network_quote_community":   self.get_quote_network_community,
            "network_reply_community":   self.get_reply_network_community,
            "network_retweet_community": self.get_retweet_network_community,
            "network_friend_community": self.get_friend_network_community,
            "userinfobio"       : self.get_user_info_bio,
            "tweetinforetweet"  : self.get_tweet_info_retweet,
            "tweetinfocreateat" : self.get_tweet_info_created_at
        }

        return

    def get_availablefeaturetypes(self):
        """
        Returns un array containing the keyword corresponding to available feature types
        :return: un array containing the keyword corresponding to available feature types
        """
        return np.array([ x for x in self.global_feature_types_list.keys()])




    #features extractor
    def create_feature_space(self,tweets,feature_types_list=None,tweet_test=None):
        """
        Create a combined feature space staking the feature space of each feature type
        :param tweets: Array of  tweet objects belonging to the training set
        :param feature_types_list: Optional. array of keyword corresponding to the keys of the dictionary global_feature_types_list
            If not defined, all available features are used
            You could add new features in global_feature_types_list. See def __init__(self).
        :param tweet_test: Optional. Array of tweet objects belonging to the test set
        :return:
            X: The feature space of the training set
            X_test: The feature space of the test set, if test  set was defined
            feature_names: An array containing the names of the feature types used for creating the combined feature space
            feature_type_indexes: A numpy array of length len(feature_type).
                               Given feature_type i, feature_type_indexes[i] contains
                               the list of the index columns of the combined feature space matrix for the feature type i

            How to use the output, example:

            feature_type=feature_manager.get_avaiblefeaturetypes()

            print(feature_type)  # available feature types
            [
              "puntuactionmarks",
              "length",
              "numhashtag",
            ]
            print(feature_name)  # the name of all feature corresponding to the  number of columns of X
            ['feature_exclamation', 'feature_question',
            'feature_period', 'feature_comma', 'feature_semicolon','feature_overall',
            'feature_charlen', 'feature_avgwordleng', 'feature_numword',
            'feature_numhashtag' ]

            print(feature_type_indexes)
            [ [0,1,2,3,4,5],
              [6,7,8],
              [9]
            ]

            print(X) #feature space 3X10 using "puntuactionmarks", "length", and "numhashtag"
            numpy.array([
            [0,1,0,0,0,1,1,0,0,1], # vector rapresentation of the document 1
            [0,1,1,1,0,1,1,0,0,1], # vector rapresentation of the document 2
            [0,1,0,1,0,1,0,1,1,1], # vector rapresentation of the document 3

            ])

            # feature space 3X6 obtained using only "puntuactionmarks"
            print(X[:, feature_type_indexes[feature_type.index("puntuactionmarks")]])
            numpy.array([
            [0,1,0,0,0,1], # vector representation of the document 1
            [0,1,1,1,0,1], # vector representation of the document 2
            [0,1,0,1,0,1], # vector representation of the document 3

            ])

        """

        #If feature_types_list is not defined, all available features are used
        if feature_types_list is None:
            feature_types_list=self.get_availablefeaturetypes()


        #Feature space obtained staking the feature space of each feature type
        if tweet_test is None:
            combined_feature_names=[]
            combined_feature_index=[]
            combined_X=[]
            index=0
            for key in feature_types_list:
                X,feature_names=self.global_feature_types_list[key](tweets,tweet_test)

                current_feature_index=[]
                for i in range(0,len(feature_names)):
                    current_feature_index.append(index)
                    index+=1
                combined_feature_index.append(current_feature_index)

                combined_feature_names=np.concatenate((combined_feature_names,feature_names))
                if combined_X!=[]:
                    combined_X=csr_matrix(hstack((combined_X,X)))
                else:
                    combined_X=X

            return combined_X, combined_feature_names, np.array(combined_feature_index)
        else:
            combined_feature_names=[]
            combined_feature_index=[]
            combined_X=[]
            combined_X_test=[]
            index=0
            for key in feature_types_list:
                X,X_test,feature_names=self.global_feature_types_list[key](tweets,tweet_test)
                current_feature_index=[]
                for i in range(0,len(feature_names)):
                    current_feature_index.append(index)
                    index+=1
                combined_feature_index.append(current_feature_index)

                combined_feature_names=np.concatenate((combined_feature_names,feature_names))
                if combined_X!=[]:
                    combined_X=csr_matrix(hstack((combined_X,X)))
                    combined_X_test=csr_matrix(hstack((combined_X_test,X_test)))
                else:
                    combined_X=X
                    combined_X_test=X_test

            return combined_X, combined_X_test, combined_feature_names, np.array(combined_feature_index)


    def get_unigram_features(self, tweets, tweet_test=None):
        """
        
        :param tweets: Array of  Tweet objects. Training set.
        :param tweet_test: Optional Array of  Tweet objects. Test set.
        :return:
        
        X_train: The feature space of the training set
        X_test: The feature space of the test set, if test  set was defined
        feature_names:  An array containing the names of the features used  for  creating the feature space
        """
        # CountVectorizer return a numpy matrix
        # row number of tweets
        # column number of 1gram in the dictionary

        countVectorizer = CountVectorizer(ngram_range=(1,1),
                                          analyzer="word",
                                          #stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)

        if tweet_test is None:
            feature  = []
            for tweet in tweets:

                feature.append(tweet.text)

            countVectorizer = countVectorizer.fit(feature)

            X = countVectorizer.transform(feature)

            feature_names=countVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature  = []
            feature_test  = []
            for tweet in tweets:

                feature.append(tweet.text)

            for tweet in tweet_test:

                feature_test.append(tweet.text)


            countVectorizer = countVectorizer.fit(feature)

            X_train = countVectorizer.transform(feature)
            X_test = countVectorizer.transform(feature_test)

            feature_names=countVectorizer.get_feature_names()

            return X_train, X_test, feature_names



    def get_ngram_1_3_features(self, tweets, tweet_test=None):
        """

        :param tweets: Array of  Tweet objects. Training set.
        :param tweet_test: Optional Array of  Tweet objects. Test set.
        :return:

        X_train: The feature space of the training set
        X_test: The feature space of the test set, if test  set was defined
        feature_names:  An array containing the names of the features used  for  creating the feature space
        """
        # CountVectorizer return a numpy matrix
        # row number of tweets
        # column number of 1-3gram in the dictionary

        countVectorizer = CountVectorizer(ngram_range=(1,3),
                                          analyzer="word",
                                          #stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)

        if tweet_test is None:
            feature  = []
            for tweet in tweets:

                feature.append(tweet.text)

            countVectorizer = countVectorizer.fit(feature)

            X = countVectorizer.transform(feature)

            feature_names=countVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature  = []
            feature_test  = []
            for tweet in tweets:

                feature.append(tweet.text)

            for tweet in tweet_test:

                feature_test.append(tweet.text)


            countVectorizer = countVectorizer.fit(feature)

            X_train = countVectorizer.transform(feature)
            X_test = countVectorizer.transform(feature_test)

            feature_names=countVectorizer.get_feature_names()

            return X_train, X_test, feature_names


    def get_ngramshashtag_features(self, tweets,tweet_test=None):
        """
        
        :param tweets: Array of  Tweet objects. Training set.
        :param tweet_test: Optional Array of  Tweet objects. Test set.
        :return:
        
        X_train: The feature space of the training set
        X_test: The feature space of the test set, if test  set was defined
        feature_names:  An array containing the names of the features used  for  creating the feature space
        """
        # CountVectorizer return a numpy matrix
        # row number of tweets
        # column number of hashtags in the dictionary

        countVectorizer = CountVectorizer(ngram_range=(1,1),
                                          analyzer="word",
                                          #stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)

        if tweet_test is None:
            feature  = []
            for tweet in tweets:

                feature.append(' '.join(re.findall(r"#(\w+)", tweet.text)))


            countVectorizer = countVectorizer.fit(feature)

            X = countVectorizer.transform(feature)

            feature_names=countVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature  = []
            feature_test  = []
            for tweet in tweets:

                feature.append(' '.join(re.findall(r"#(\w+)", tweet.text)))

            for tweet in tweet_test:

                feature_test.append(' '.join(re.findall(r"#(\w+)", tweet.text)))


            countVectorizer = countVectorizer.fit(feature)

            X_train = countVectorizer.transform(feature)
            X_test = countVectorizer.transform(feature_test)

            feature_names=countVectorizer.get_feature_names()

            return X_train, X_test, feature_names


    def get_nchargrams_features(self, tweets,tweet_test=None):

        # CountVectorizer return a numpy matrix
        # row number of tweets
        # column number of 2-5chargrams in the dictionary
        countVectorizer = CountVectorizer(ngram_range=(2,5),
                                          analyzer="char",
                                          #stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)

        if tweet_test is None:
            feature  = []
            for tweet in tweets:

                feature.append(tweet.text)


            countVectorizer = countVectorizer.fit(feature)

            X = countVectorizer.transform(feature)

            feature_names=countVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature  = []
            feature_test  = []
            for tweet in tweets:

                feature.append(tweet.text)

            for tweet in tweet_test:

                feature_test.append(tweet.text)


            countVectorizer = countVectorizer.fit(feature)

            X_train = countVectorizer.transform(feature)
            X_test = countVectorizer.transform(feature_test)

            feature_names=countVectorizer.get_feature_names()

            return X_train, X_test, feature_names


    def get_numhashtag_features(self, tweets,tweet_test=None):

        # This method extracts a single column: the number of hashtags in the tweet
        # len(tweets) x 1
        # sr_matrix(np.vstack(feature)) convert to an array of dimension len(tweets)X1

        if tweet_test is None:
            feature  = []

            for tweet in tweets:
                feature.append(len(re.findall(r"#(\w+)", tweet.text)))

            return csr_matrix(np.vstack(feature)),\
                   ["FeatureNumHashtag"]

        else:
            feature  = []
            feature_test  = []

            for tweet in tweets:
                feature.append(len(re.findall(r"#(\w+)", tweet.text)))

            for tweet in tweet_test:
                feature_test.append(len(re.findall(r"#(\w+)", tweet.text)))

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_test)),\
                   ["FeatureNumHashtag"]


    def get_puntuaction_marks_features(self,tweets,tweet_test):

        # This method extracts six columns corresponding to the number of exclamations, question marks, periods, commas, semicolon, general puntuaction marks.
        # len(tweets) x 6
        # sr_matrix(np.vstack(feature)) convert to an array of dimension len(tweets)X6

        if tweet_test is None:
            feature  = []

            for tweet in tweets:
                feature.append([
                len(re.findall(r"[!]", tweet.text)),
                len(re.findall(r"[?]", tweet.text)),
                len(re.findall(r"[.]", tweet.text)),
                len(re.findall(r"[,]", tweet.text)),
                len(re.findall(r"[;]", tweet.text)),
                len(re.findall(r"[!?.,;]", tweet.text)),
                ]
            )
            return csr_matrix(np.vstack(feature)),\
                   ["FeaturePuntuactionMarksExclamation",
                    "FeaturePuntuactionMarksQuestion",
                    "FeaturePuntuactionMarksPeriod",
                    "FeaturePuntuactionMarksComma",
                    "FeaturePuntuactionMarksSemicolon",
                    "FeaturePuntuactionMarksOverall"]
        else:
            feature  = []
            feature_test  = []

            for tweet in tweets:
                feature.append([
                len(re.findall(r"[!]", tweet.text)),
                len(re.findall(r"[?]", tweet.text)),
                len(re.findall(r"[.]", tweet.text)),
                len(re.findall(r"[,]", tweet.text)),
                len(re.findall(r"[;]", tweet.text)),
                len(re.findall(r"[!?.,;]", tweet.text)),
                ]

            )

            for tweet in tweet_test:
                feature_test.append([
                len(re.findall(r"[!]", tweet.text)),
                len(re.findall(r"[?]", tweet.text)),
                len(re.findall(r"[.]", tweet.text)),
                len(re.findall(r"[,]", tweet.text)),
                len(re.findall(r"[;]", tweet.text)),
                len(re.findall(r"[!?.,;]", tweet.text)),
                ]

            )
            print(len(feature),len(feature_test))
            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_test)),\
                   ["FeaturePuntuactionMarksExclamation",
                    "FeaturePuntuactionMarksQuestion",
                    "FeaturePuntuactionMarksPeriod",
                    "FeaturePuntuactionMarksComma",
                    "FeaturePuntuactionMarksSemicolon",
                    "FeaturePuntuactionMarksOverall"]


    def get_length_features(self,tweets,tweet_test):

        # This method extracts 3 columns corresponding to the number of char, the number od token, the average lenght of the tokens in each tweet
        # len(tweets) rows of 3 columns
        # sr_matrix(np.vstack(feature)) convert to an array of dimension len(tweets)X3

        if tweet_test is None:
            feature  = []

            for tweet in tweets:
                feature.append([
                len(tweet.text),
                int(np.average([len(w) for w in tweet.text.split(" ")])),
                len(tweet.text.split(" ")),

                ]

            )


            return csr_matrix(np.vstack(feature)),\
                   ["FeatureLengthChar",
                    "FeatureLengthAverageWord",
                    "FeatureLengthWord"]

        else:
            feature  = []
            feature_test  = []

            for tweet in tweets:
                feature.append([
                len(tweet.text),
                int(np.average([len(w) for w in tweet.text.split(" ")])),
                len(tweet.text.split(" ")),

                ]

            )


            for tweet in tweet_test:
                feature_test.append([
                len(tweet.text),
                int(np.average([len(w) for w in tweet.text.split(" ")])),
                len(tweet.text.split(" ")),

                ]

            )


            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_test)),\
                   ["FeatureLengthChar",
                    "FeatureLengthAverageWord",
                    "FeatureLengthWord"]


    def get_quote_network_community(self,tweets,tweet_test):

        # This method extracts a column for each community in the network
        # len(tweets) rows and M columns corresponding to the number of distinct communities  (+ 1, for the disconnected vertex)

        countVectorizer = CountVectorizer(ngram_range=(1, 1),
                                          analyzer="word",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)
        if tweet_test is None:
            feature = []
            for tweet in tweets:
                feature.append("FeatureQuoteNetworkCommunity"+str(tweet.community_quote))

            countVectorizer = countVectorizer.fit(feature)

            X = countVectorizer.transform(feature)

            feature_names = countVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                feature.append("FeatureQuoteNetworkCommunity"+str(tweet.community_quote))

            for tweet in tweet_test:
                feature_test.append("FeatureQuoteNetworkCommunity"+str(tweet.community_quote))

            countVectorizer = countVectorizer.fit(feature)

            X_train = countVectorizer.transform(feature)
            X_test = countVectorizer.transform(feature_test)

            feature_names = countVectorizer.get_feature_names()

            return X_train, X_test, feature_names

    def get_reply_network_community(self,tweets,tweet_test):

        # This method extracts a column for each community in the network
        # len(tweets) rows and M columns corresponding to the number of distinct communities  (+ 1, for the disconnected vertex)

        countVectorizer = CountVectorizer(ngram_range=(1, 1),
                                          analyzer="word",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)
        if tweet_test is None:
            feature = []
            for tweet in tweets:
                feature.append("FeatureReplyNetworkCommunity"+str(tweet.community_reply))

            countVectorizer = countVectorizer.fit(feature)

            X = countVectorizer.transform(feature)

            feature_names = countVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                feature.append("FeatureReplyNetworkCommunity"+str(tweet.community_reply))

            for tweet in tweet_test:
                feature_test.append("FeatureReplyNetworkCommunity"+str(tweet.community_reply))

            countVectorizer = countVectorizer.fit(feature)

            X_train = countVectorizer.transform(feature)
            X_test = countVectorizer.transform(feature_test)

            feature_names = countVectorizer.get_feature_names()

            return X_train, X_test, feature_names



    def get_friend_network_community(self,tweets,tweet_test):

        # This method extracts a column for each community in the network
        # len(tweets) rows and M columns corresponding to the number of distinct communities  (+ 1, for the disconnected vertex)

        countVectorizer = CountVectorizer(ngram_range=(1, 1),
                                          analyzer="word",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)
        if tweet_test is None:
            feature = []
            for tweet in tweets:
                feature.append("FeatureFriendNetworkCommunity"+str(tweet.community_friend))

            countVectorizer = countVectorizer.fit(feature)

            X = countVectorizer.transform(feature)

            feature_names = countVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                feature.append("FeatureFriendNetworkCommunity"+str(tweet.community_friend))

            for tweet in tweet_test:
                feature_test.append("FeatureFriendNetworkCommunity"+str(tweet.community_friend))

            countVectorizer = countVectorizer.fit(feature)

            X_train = countVectorizer.transform(feature)
            X_test = countVectorizer.transform(feature_test)

            feature_names = countVectorizer.get_feature_names()

            return X_train, X_test, feature_names

    def get_retweet_network_community(self,tweets,tweet_test):

        # This method extracts a column for each community in the network
        # len(tweets) rows and M columns corresponding to the number of distinct communities  (+ 1, for the disconnected vertex)

        countVectorizer = CountVectorizer(ngram_range=(1, 1),
                                          analyzer="word",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)
        if tweet_test is None:
            feature = []
            for tweet in tweets:
                feature.append("FeatureRetweetNetworkCommunity"+str(tweet.community_retweet))

            countVectorizer = countVectorizer.fit(feature)

            X = countVectorizer.transform(feature)

            feature_names = countVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                feature.append("FeatureRetweetNetworkCommunity"+str(tweet.community_retweet))

            for tweet in tweet_test:
                feature_test.append("FeatureRetweetNetworkCommunity"+str(tweet.community_retweet))

            countVectorizer = countVectorizer.fit(feature)

            X_train = countVectorizer.transform(feature)
            X_test = countVectorizer.transform(feature_test)

            feature_names = countVectorizer.get_feature_names()

            return X_train, X_test, feature_names

    def get_user_info_bio(self,tweets,tweet_test):

        # This method extracts a column for each emoji in the corpus
        # len(tweets) rows and M (number of distinct emoji)

        countVectorizer = CountVectorizer(ngram_range=(1, 1),
                                          analyzer="word",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)
        if tweet_test is None:
            feature = []
            for tweet in tweets:
                text='FeatureUserInfoBio '
                if len(tweet.user_info['emojy_in_bio'])>0:
                    for emoji in  tweet.user_info['emojy_in_bio'].split(":--:"):
                        if len(emoji.split("---->")) == 2:
                            text += "     FeatureUserInfoBio" + re.sub("[^a-zA-Z0-9]","",emoji.split("---->")[1].strip())
                feature.append(text)

            countVectorizer = countVectorizer.fit(feature)

            X = countVectorizer.transform(feature)

            feature_names = countVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                text='FeatureUserInfoBio '
                if tweet.user_info['emojy_in_bio'] is not None:
                    for emoji in tweet.user_info['emojy_in_bio'].split(":--:"):
                        if len(emoji.split("---->")) == 2:
                            text += "     FeatureUserInfoBio" + re.sub("[^a-zA-Z0-9]","",emoji.split("---->")[1].strip())
                feature.append(text)

            for tweet in tweet_test:
                text = 'FeatureUserInfoBio '
                if tweet.user_info['emojy_in_bio'] is not None:
                    for emoji in tweet.user_info['emojy_in_bio'].split(":--:"):
                        if len(emoji.split("---->")) == 2:
                            text += "     FeatureUserInfoBio" + re.sub("[^a-zA-Z0-9]","",emoji.split("---->")[1].strip())
                feature_test.append(text)

            countVectorizer = countVectorizer.fit(feature)

            X_train = countVectorizer.transform(feature)
            X_test = countVectorizer.transform(feature_test)

            feature_names = countVectorizer.get_feature_names()
            return X_train, X_test, feature_names


    def get_tweet_info_retweet(self, tweets,tweet_test=None):

        # This method extracts a single column correspondig to the number of retweet that the tweet received
        # len(tweets) rows of 1 column

        if tweet_test is None:
            feature  = []

            for tweet in tweets:
                feature.append(tweet.tweet_info['retweet_count'])

            return csr_matrix(np.vstack(feature)),\
                   ["FeatureTweetInfoRetweet"]

        else:
            feature  = []
            feature_test  = []

            for tweet in tweets:
                feature.append(tweet.tweet_info['retweet_count'])

            for tweet in tweet_test:
                feature_test.append(tweet.tweet_info['retweet_count'])

            return csr_matrix(np.vstack(feature)),\
                   csr_matrix(np.vstack(feature_test)),\
                   ["FeatureTweetInfoRetweet"]


    def get_tweet_info_created_at(self,tweets,tweet_test):

        # This method extracts two column for each tweet: the hour and the week day of the posting time

        countVectorizer = CountVectorizer(ngram_range=(1, 1),
                                          analyzer="word",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)
        if tweet_test is None:
            feature = []
            for tweet in tweets:
                feature.append("FeatureTweetInfoCreatedAtHour"+str(datetime.strptime(tweet.tweet_info['created_at'],'%Y-%m-%d %H:%M:%S').hour)+
                               "   "+
                               "FeatureTweetInfoCreatedAtWeekDay" + str(datetime.strptime(tweet.tweet_info['created_at'],'%Y-%m-%d %H:%M:%S').weekday())
                               )

            countVectorizer = countVectorizer.fit(feature)

            X = countVectorizer.transform(feature)

            feature_names = countVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                feature.append("FeatureTweetInfoCreatedAtHour"+str(datetime.strptime(tweet.tweet_info['created_at'],'%Y-%m-%d %H:%M:%S').hour)+
                               "   "+
                               "FeatureTweetInfoCreatedAtWeekDay" + str(datetime.strptime(tweet.tweet_info['created_at'],'%Y-%m-%d %H:%M:%S').weekday())
                               )
            for tweet in tweet_test:
                feature_test.append("FeatureTweetInfoCreatedAtHour"+str(datetime.strptime(tweet.tweet_info['created_at'],'%Y-%m-%d %H:%M:%S').hour)+
                               "   "+
                               "FeatureTweetInfoCreatedAtWeekDay" + str(datetime.strptime(tweet.tweet_info['created_at'],'%Y-%m-%d %H:%M:%S').weekday())
                               )
            countVectorizer = countVectorizer.fit(feature)

            X_train = countVectorizer.transform(feature)
            X_test = countVectorizer.transform(feature_test)

            feature_names = countVectorizer.get_feature_names()

            return X_train, X_test, feature_names



#inizializer
def make_feature_manager():

    features_manager = Features_manager()

    return features_manager
