import praw 
import json
import pandas as pd
import requests
from datetime import datetime
client_id = 'gQiqGGqueUTkq_qzsbjirA'
secret = 'xL2ovMlrvgEzxcX1KeifLYzWNPMIqQ'
user_agent = 'Scraping'

#Reddit login
reddit = praw.Reddit(client_id=client_id, client_secret=secret, user_agent=user_agent)

class reddit_scarping_random():
    def unix_time_to_date(self):
        """Self should be a string of unix time 
        
        """
        ts = int(self)
        try:
            new = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            return str(new)
        except:
            print("UNIX TIME CONVERSION ERROR")
        
        
    def get_reddit(self, subreddit, count, timeframe, listing):
        """
            timeframe = 'day' #hour, day, week, month, year, all
            listing = 'random' # controversial, best, hot, new, random, rising, top
 
        """
        try:
            base_url = f'https://www.reddit.com/r/{subreddit}/{listing}.json?count={count}&t={timeframe}'
            request = requests.get(base_url, headers = {'User-agent': 'yourbot'})
        except:
            print('An Error Occured')
        return request.json()
        
    def get_reddit_by_listing(self, subreddit, count, timeframe, listing):
        """
            timeframe = 'day' #hour, day, week, month, year, all
            listing = 'random' # controversial, best, hot, new, random, rising, top
        """
        overall_posts = []
        posts = []
        top_post = get_reddit(subreddit,count,timeframe, listing)
        if listing != 'random':
            for i in range(count):
                title = top_post['data']['children'][i]['data']['title']
                subreddit_ = top_post['data']['children'][i]['data']['subreddit']
                score = top_post['data']['children'][i]['data']['score']
                url = top_post['data']['children'][i]['data']['url']
                time_created = reddit_scarping_random.unix_time_to_date(str(top_post['data']['children'][i]['data']['created']))
                body = top_post['data']['children'][i]['data']['selftext']
                posts.append([title, subreddit_, score, url, body, time_created])
            overall_posts.append(posts)
            overall_posts_ = pd.DataFrame(posts, columns=['title', 'subreddit', 'score', 'url', 'body', 'time_created' ])
            overall_posts_.to_csv("TrendPosting.csv")
        else: #needs to continue development matching above 
            title_ = top_post[0]['data']['children'][0]['data']['title']
            url_ = top_post[0]['data']['children'][0]['data']['url']
        
        
    def get_comment_by_key_word(self):
        """Self should be a list of string containing the key words
        
        """
        overall_posts = []
        posts = []
        
        for key_word in self:
            subreddit_ = reddit.subreddit(key_word)
            try:
                for post in subreddit_.hot(limit=100000):
                    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
                #posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])
                overall_posts.append(posts)
            except:
                print("Key word not found!")
        overall_posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])
        overall_posts.to_csv("information.csv")

#Test code - test passed
if __name__ == "__main__":
    #reddit_scarping.get_comment_by_key_word(['credit', 'finance']) 
    print(reddit_scarping_random.unix_time_to_date("1284101485"))