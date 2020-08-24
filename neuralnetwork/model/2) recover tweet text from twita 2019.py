__author__ = 'mirko'
import oauth2 as oauth
import time
import json
import csv
from datetime import datetime

CONSUMER_KEY    = ""
CONSUMER_SECRET = ""
ACCESS_KEY      = ""
ACCESS_SECRET   = ""


consumer = oauth.Consumer(key=CONSUMER_KEY, secret=CONSUMER_SECRET)
access_token = oauth.Token(key=ACCESS_KEY, secret=ACCESS_SECRET)
client = oauth.Client(consumer, access_token)


idList=[]
infile = open("twita-2019/twita-2019-recovered.csv", 'r')
spamreader = csv.reader(infile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
for row in spamreader:
    idList.append(row[0])


print(len(idList))

while len(idList) > 0:
    parameter = ','.join(idList[0:100]) #max 100 id per request
    try:
        lookup_endpoint ="https://api.twitter.com/1.1/statuses/lookup.json?id="+parameter
        response, data = client.request(lookup_endpoint)
        if response['status']=='200' and  'x-rate-limit-remaining' in response and  'x-rate-limit-reset' in response:

            idList[0:100] = []
            jsonTweet=json.loads(data)

            outfile=open('twita-2019/twita-2019.txt','a')

            for tweet in jsonTweet:

                if 'timestamp_ms' in tweet:
                    date = datetime.fromtimestamp(int(tweet['timestamp_ms']) / 1000)
                elif 'created_at' in tweet:
                    date = datetime.strptime(tweet["created_at"], '%a %b %d %H:%M:%S %z %Y')

                outfile.write(tweet['text']+"\n")

            outfile.close()

            print('id rescue: wait '+str((15*60)/int(response['x-rate-limit-limit']))+' seconds')
            time.sleep((15*60)/int(response['x-rate-limit-limit']))

        else:
            print(response)
            print(data)
            time.sleep(60)
    except Exception as e:
        print('Error:',e)
        time.sleep(60)

