__author__ = 'mirko'
import csv

class User_Info(object):
    users={}
    def __init__(self):

            """Return an array containing users.
               users are encoded as objects.
            """
            self.users = {}
            csvfile = open("data/USER.csv")
            next(csvfile)  # skip header
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for user in spamreader:
                # "user_id", "statuses_count", "friends_count", "followers_count", "listed_count", "created_at", "emojy_in_bio"

                self.users[user[0]] = {}
                self.users[user[0]]['statuses_count'] = int(user[1])
                self.users[user[0]]['friends_count'] = int(user[2])
                self.users[user[0]]['followers_count'] = int(user[3])
                self.users[user[0]]['listed_count'] = int(user[4])
                self.users[user[0]]['created_at'] = user[5]
                self.users[user[0]]['emojy_in_bio'] = user[6]



    def get_user_info(self,user_id):
        return self.users[user_id]

def make_user_info():
    user_info = User_Info()

    return user_info
