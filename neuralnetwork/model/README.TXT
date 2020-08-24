The file tweets_2019.txt, containing the text of all tweets gathered using the Twitter's Stream API during the year 2019 (about 16M tweets discarding retweets, quotes, replies, and tweets containing urls or media), was used for training the released word2vect model.

We can not share the textual content of these tweets (https://developer.twitter.com/en/developer-terms/agreement-and-policy)
For this reason, we initially release only the pre-trained word2vect model.

We decided to release the list of the ids gathered by TWITA (Basile et. al (2018) Long-term Social Media Data Collection at the University of Turin) in 2019 after your request.

For doing that we release the file tweet_id-2019-compressed.zip.
The file tweet_id-2019-compressed.zip  is encrypted. Please register to our Google Group (https://groups.google.com/g/sardistance-evalita2020) and compile this Google Form regarding Copyright Agreement, to obtain the passwords.
The file tweet_id-2019-compressed.csv contains the list of 18M (new old tweets was recovered in the last month of the current year) gathered from TWITA discarding retweets, quotes, replies, and tweets containing urls or media.

Github requires that the limit size of a file is 100Mb.
Therefore, for saving space we did not store the id of each tweet, but only the numerical difference between each pair of two consecutive ids.

The file "1) recover tweet id from twita 2019.py" recovers the ids list. twita-2019-recovered.csv is the output file.

The file "2) recover tweet text from twita 2019.py" recovers the json of each listed tweet id using the Twitter's API. twita-2019.txt is the output file.

twita-2019.txt could be used for training other types of models.

I wish you good work
Regards,
Sardistance team
