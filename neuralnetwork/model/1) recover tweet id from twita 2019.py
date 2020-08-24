__author__ = 'mirko'
# -*- coding: utf-8 -*-
import csv

csvfileread=open('twita-2019/tweet_id-2019-compressed.csv', 'r', newline='')
spamreader = csv.reader(csvfileread, delimiter=',',
                    quotechar='"', quoting=csv.QUOTE_MINIMAL)

csvfile=open('twita-2019/twita-2019-recovered.csv', 'w', newline='')
spamwriter = csv.writer(csvfile, delimiter=',',
                    quotechar='"', quoting=csv.QUOTE_MINIMAL)

last=0
first=True
for tweet in spamreader:
    if first:
        first=False
        continue
    gap=int(tweet[0])

    decompressed_tweet_id = gap + last
    last=decompressed_tweet_id
    spamwriter.writerow([decompressed_tweet_id])

csvfileread.close()
csvfile.close()
