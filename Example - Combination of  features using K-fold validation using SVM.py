from itertools import combinations
import numpy
from sklearn.svm.classes import SVC
from machinelearning import Features_manager
from utils import Database_manager
from sklearn.metrics.classification import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import KFold
"""
You could use this script for evaluating your feature using the K-fold validation on training set.
The script is able to test all the combination of feature types for finding the best result

"""
# initialize database_manager
database_manager=Database_manager.make_database_manager()
# initialize feature_manager
feature_manager=Features_manager.make_feature_manager()

# recover training tweets
tweets=numpy.array(database_manager.return_tweets_training())
labels=numpy.array(database_manager.get_label(tweets))

# recover keyword list corresponding to available features
feature_types=feature_manager.get_availablefeaturetypes()

# create the feature space with all available features
X,feature_names,feature_type_indexes=feature_manager.create_feature_space(tweets, feature_types)

print("feature space dimension X:", X.shape)
"""
https://en.wikipedia.org/wiki/Combination
"""
N = len(feature_types)
for K in range(1, N):
    for subset in combinations(range(0, N), K):
        # it extracts the columns of the features considered in the current combination
        # the feature space is reduced
        feature_index_filtered = numpy.array([list(feature_types).index(f) for f in feature_types[list(subset)]])
        feature_index_filtered = numpy.concatenate(feature_type_indexes[list(feature_index_filtered)])

        X_filter = X[:,feature_index_filtered]
        kf = KFold(n_splits=5, random_state=True)
        golden = []
        predict = []
        for index_train, index_test in kf.split(X):

            X_train = X_filter[index_train]
            X_test  = X_filter[index_test]
            clf= SVC(kernel='linear')
            clf.fit(X_train, labels[index_train])
            test_predict = clf.predict(X_test)

            golden = numpy.concatenate((golden, labels[index_test]), axis=0)
            predict = numpy.concatenate((predict, test_predict), axis=0)

        prec, recall, f, support = precision_recall_fscore_support(
            golden,
            predict,
            beta=1)

        accuracy = accuracy_score(
            golden,
            predict
        )
        #it prints the metrics for the current combination of features
        print(feature_types[list(subset)])
        print("feature space dimention X:", X_filter.shape)
        #print(prec, recall, f, support )
        print("f[0]+f[1])/2",(f[0]+f[1])/2)
        print("(f[0]+f[1]+f[2])/3",(f[0]+f[1]+f[2])/3)
        #print(accuracy)


