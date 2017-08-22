import pandas as pd
import numpy as np
import json
import re
import requests
import string
import urllib, json
import nltk
import urllib2
import time
#from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing.pool import ThreadPool
tweets_data_path = 'real_tweets.json'
tweets_data = []
tweets_file = open(tweets_data_path, "r")

#f= open('itemfeaturestestingtoday2.txt', 'a') 
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue     
    tweets = pd.DataFrame()
    
    def getTweetId(tweetsara):
        twId = map(lambda tweet: tweet['id_str'], tweetsara)
        for i in twId:
            twIdStr = i.encode('utf-8', 'ignore')
        #postTextStr = ''.join(str(i) for i in postText)
        return twIdStr
    
    def checkForExternalLinks(tweets_data):
        numurls1 = []
        urlList = []
        numUrls1 = map(lambda tweet: tweet['entities']['urls'] if tweet['entities'] != None else None, tweets_data)
        if len(numUrls1) > 0:
            numurls1.extend(tweet['entities']['urls'])
            tweet['url1'] = [numurl['url'] for numurl in numurls1]
            #print tweet['url1']
            urlList.extend(tweet['url1'])
        #print urlList
        for ural in urlList:
            checkForImage = isAnImage(ural)
            if  not checkForImage:
                return ural
            else:
                return None
    
    def organizeRunRank(tweetid, extLink, filePath):
    
        result = None
        numberThreads = 114;
        pool = ThreadPool(4) 
        #pool = ThreadPool(processes=1)

        for i in range (0, numberThreads):
            results = pool.apply_async(RunnableRank, (i, extLink, filePath))

        try:
            for i in range(0, numberThreads):
                val = results.get() 
                if (val!= None):
                    result = val
                    pool.terminate()

        except KeyboardInterrupt:
            result = 0

        return result

    def RunnableRank(name, url, filePath):
        threadName = name
        urlName = url
        filePathName = filePath
        rank = None
        new_dict = {}
        
        textfile = open(filePath, 'r')
        filetext = textfile.readlines()
        textfile.close()
        for i in filetext:
            parts = i.split("\t")[0:1]
            new_dict(parts[0], parts[1])
            
        strRank = new_dict.get(urlName, default=None)
        if (strRank != None):
            rank = float(strRank);
        new_dict.clear()
        return rank
      
    tweetid = str(getTweetId(tweets_data))
    print tweetid
    externLink = checkForExternalLinks(tweets_data)
    indegree = organizeRunRank(tweetid,externLink,"D:/Downloads/hostgraph-indegree.tsv/hostgraph-indegree.tsv")
    harmonic = None
    
    if indegree != None:
        harmonic = organizeRunRank(tweetid, externLink, "D:/Downloads/hostgraph-h.tsv/hostgraph-h.tsv")
    
    print indegree
    print harmonic
