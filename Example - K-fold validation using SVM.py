import numpy
from sklearn.svm.classes import SVC
from machinelearning import Features_manager
from utils import Database_manager
from sklearn.metrics.classification import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import KFold
"""
You could use this script for evaluating your feature using the K-fold validation on training set

"""


# initialize database_manager
database_manager = Database_manager.make_database_manager()
# initialize feature_manager
feature_manager = Features_manager.make_feature_manager()

# recover tweets
tweets = numpy.array(database_manager.return_tweets_training())
labels = numpy.array(database_manager.get_label(tweets))

# recover keyword list corresponding to available features
feature_types = feature_manager.get_availablefeaturetypes()
"""
or you could include only desired features
feature_types=[
            "unigram",
            "unigramhashtag"
            ]
"""
feature_types=[
            "unigram",
            ]
# create the feature space with all available features
X,feature_names,feature_type_indexes=feature_manager.create_feature_space(tweets,feature_types)


print("features:", feature_types)
print("feature space dimension:", X.shape)

golden=[]
predict=[]
kf = KFold(n_splits=5, random_state=True)
for index_train, index_test in kf.split(X):

    clf = SVC(kernel="linear")

    clf.fit(X[index_train],labels[index_train])
    test_predict = clf.predict(X[index_test])

    golden=numpy.concatenate((golden,labels[index_test]), axis=0)
    predict=numpy.concatenate((predict,test_predict), axis=0)

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
print(support)
print(accuracy)
print("f[0]+f[1])/2", (f[0] + f[1]) / 2)
