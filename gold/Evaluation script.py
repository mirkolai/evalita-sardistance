import csv
from sklearn.metrics import  precision_recall_fscore_support


y_true= {}
csvfile=open('TEST-GOLD.csv')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
for row in spamreader:
    if row[1] in ['AGAINST','FAVOR','NONE']:
        y_true[row[0]]=row[1].lower()
print("TEST-GOLD: ",len(y_true))

y_pred={}
csvfile=open('TEST_PREDICTION-baseline-SVC.tsv')
spamreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
for row in spamreader:
    if row[1] in ['AGAINST','FAVOR','NONE']:
        y_pred[row[0]]=row[1].lower()
print("TEST-PRED: ",len(y_pred))

l_true=[]
l_pred=[]
for tweet_id in y_true.keys():
    l_true.append(y_true[tweet_id])
    l_pred.append(y_pred[tweet_id])

if len(y_pred) == len(y_true):
    prec, recall, f, support = precision_recall_fscore_support(l_true, l_pred, average=None)

print((f[0]+f[1])/2,prec, recall, f, support)

