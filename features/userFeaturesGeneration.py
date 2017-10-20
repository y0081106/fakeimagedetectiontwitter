import numpy as np
import json
import re
import requests
import string
import urllib, json
import urllib2
import time
import csv
import nltk
import pickle
import pandas as pd
import emoji
import datetime as dt

from bs4 import BeautifulSoup
from PIL import Image
from textstat.textstat import textstat
from datetime import datetime
from dateutil.parser import parse
from geopy.geocoders import Nominatim
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from langdetect import detect
from collections import Counter



start_time = time.time()
tweets_data_path = 'C:/Users/imaad/twitteradvancedsearch/fake_real_tweets_training.json'
tweets_data = []
ufl_result = []
tweet_id = [] 
numFriends = []
numFollowers = []
timesListed = []
followerFriendRatio = []
hasUrlCheck = []
userUrl = []
bioCheck = [] 
verifiedUser,numFavorites = [],[]
locationCheck, existingLocationCheck = [],[]
profileImgCheck, headerImgCheck, accountAge, tweetRatio, mediaContent = [],[],[],[],[]
indegreeval, harmonicval,AlexaPopularity, AlexaReach, AlexaDelta,AlexaCountry, WotValue  = [],[],[],[],[],[],[]
tweets_file = open(tweets_data_path, "r")
Indegree, Harmonic = [],[]

for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue
#print len(tweets_data)    
    tweets = pd.DataFrame()
    
    def getTweetId(tweetsara):
        twId = map(lambda tweet: tweet['id_str'], tweetsara)
        for i in twId:
            twIdStr = i.encode('utf-8', 'ignore')
        #postTextStr = ''.join(str(i) for i in postText)
        return twIdStr

    def getNumFriends(tweetsara):
        numFriends = map(lambda tweet: tweet['user']['friends_count'], tweetsara)
        #numFriendsStr = ''.join(str(i) for i in numFriends)
        for i in numFriends:
            numFriendsStr = str(i).encode('utf-8', 'ignore')
        return numFriendsStr

    def getNumFollowers(tweetsara):
        numFollowers = map(lambda tweet: tweet['user']['followers_count'], tweetsara)
        #numFollowersStr = ''.join(str(i) for i in numFollowers)
        for i in numFollowers:
            numFollowersStr = str(i).encode('utf-8', 'ignore')
        return numFollowersStr

    def getFollowerFriendRatio(numFol, numFr):
        numFol = np.array(numFol, dtype = np.float)
        numFr = np.array(numFr, dtype = np.float)
        ratio = 0
        if (numFr!= 0):
            ratio = numFol/numFr
        return ratio
    def getTimesListed(tweetsara):
        numListed = map(lambda tweet: tweet['user']['listed_count'], tweetsara)
        #numListedStr = ''.join(str(i) for i in numListed)
        for i in numListed:
            numListedStr = str(i).encode('utf-8', 'ignore')
        return numListedStr
    def hasUrl(tweetsara):
        url = map(lambda tweet: tweet['user']['url'], tweetsara)
        #print url
        if url ==[None]:
            return False
        else:
            return True
    def getUserUrl(tweetsara):
        userUrl = map(lambda tweet: tweet['user']['url'], tweetsara)
        #userUrlStr = ''.join(str(i) for i in userUrl)
        for i in userUrl:
            userUrlStr = str(i).encode('utf-8', 'ignore')
        return userUrlStr
    def hasBio(tweetsara):
        userDescription = map(lambda tweet: tweet['user']['description'], tweetsara)
        #print userDescription
        #nullValue = [x for x in userDescription if userDescription.count(x) > 1]
        #print nullValue
        if '' in userDescription:
            return False
        else:
            return True
        #return userDescription
    def isVerifiedUser(tweetsara):
        userVerification = map(lambda tweet: tweet['user']['verified'], tweetsara)
        #userVerificationStr = ''.join(str(i) for i in userVerification)
        for i in userVerification:
            userVerificationStr = str(i).encode('utf-8', 'ignore')
        return userVerificationStr
    def getNumTweets(tweetsara):
        userTweets = map(lambda tweet: tweet['user']['statuses_count'], tweetsara)
        #userTweetsStr = ''.join(str(i) for i in userTweets)
        for i in userTweets:
            userTweetsStr = str(i).encode('utf-8', 'ignore')

        return userTweetsStr
    def getNumFavorites(tweetsara):
        userFavCount = map(lambda tweet: tweet['user']['favourites_count'], tweetsara)
        #userFavCountStr = ''.join(str(i) for i in userFavCount)
        for i in userFavCount:
            userFavCountStr = str(i).encode('utf-8', 'ignore')
        
        return userFavCountStr
    def getLocation(tweetsara):
        userLoc = map(lambda tweet: tweet['user']['location'], tweetsara) 
        #userLocStr = ''.join(str(i) for i in userLoc)
        for i in userLoc:
            userLocStr = i.encode('utf-8', 'ignore')
        
        if userLocStr:
            return userLocStr
        else:
            return None
    def hasLocation(tweetsara):
        hasLoc = False
        userLoc = map(lambda tweet: tweet['user']['location'], tweetsara)
        #userLocStr = ''.join(str(i) for i in userLoc)
        for i in userLoc:
            userLocStr = i.encode('utf-8', 'ignore')
       
        if (userLocStr == "Worldwide") or len(userLocStr) != 0:
            hasLoc = True
        return hasLoc
    def hasExistingLocation(tweets_data):
        try:
            geolocator = Nominatim()
            existingLocation = geolocator.geocode(getLocation(tweets_data))
            if existingLocation == None:
                return False
            if(existingLocation.address):
                return True
            else:
                return False
        except:
            return False


    def getAlexaPopularity(domain):
        if domain == None:
            return 0
        url = "http://data.alexa.com/data?cli=10&url="+ domain
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        try:
            alexaPopularity = soup.popularity['text']
            return alexaPopularity
        except TypeError:
            return 0

    def getAlexaReachRank(domain):
        if domain == None:
            return 0
        url = "http://data.alexa.com/data?cli=10&url="+ domain
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        try:
            alexaReachRank = soup.reach['rank']
            return alexaReachRank
        except TypeError:
            return 0

    def getAlexaDeltaRank(domain):
        if domain == None:
            return 0
        url = "http://data.alexa.com/data?cli=10&url="+ domain
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        try:
            alexaDeltaRank = soup.rank['delta']
            return alexaDeltaRank
        except TypeError:
            return 0

    def getAlexaCountryRank(domain):
        if domain == None:
            return 0
        url = "http://data.alexa.com/data?cli=10&url="+ domain
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        try:
            alexaCountryRank = soup.country['rank']
            return alexaCountryRank
        except TypeError:
            return 0
    
    def isAnImage(url):
        text = 'pbs.twimg.com'
        text1 = 'p.twimg.com'
        if re.search(url, text) or re.search(url, text1):
            return True
        try:
            im = Image.open(urllib2.urlopen(url))
            if im != None:
                return True
            else:
                return False
        except IOError:
            return False

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

    def hasExternalLinks(tweets_data):
            extLinks = checkForExternalLinks(tweets_data)
            if extLinks:
                return True
            else:
                return False
            
    def expandedUrl(shortenedUrl):
        if shortenedUrl == None:
            return None
        expandedUrl = (requests.get(shortenedUrl).url)
        expandedUrlName = expandedUrl.split("/")[2:3]
        expandedUrlNameStr = str(expandedUrlName[0])
        expandedUrlNameStr = expandedUrlNameStr.replace("http://","")
        expandedUrlNameStr = expandedUrlNameStr.replace("www.", "")
        return expandedUrlNameStr

    def expandUrl(shortenedUrl):
        expandedUrl = (requests.get(shortenedUrl).url)
        #return expandedUrl
    #get the expanded Url
    #expandedUrl = expandUrl("https://t.co/GMxDlarDph")
    #print expandedUrl
    #split the name of the url
        expandedUrlName = expandedUrl.split("/")[2:3]
        expandedUrlNameStr = str(expandedUrlName[0])
        return expandedUrlNameStr

    def getWotTrustValue(host):
        if host == None:
            return 0
        expandUrlName = expandUrl(host)
        #print expandUrlName
        if "/" in expandUrlName:
            expandUrlName1 = expandUrlName.split("/")[2:3]
        else:
            expandUrlName1 = expandUrlName.split("/")[:]
        expandUrlNameStr = str(expandUrlName1[0])
        value = [None]
        url = "http://api.mywot.com/0.4/public_link_json2?hosts="+ expandUrlNameStr +"/&key=108d4b2a42ea1afc370e668b39cabdceaa19fcf0"
        #print url
        response = urllib.urlopen(url)
        data = json.load(response)
        #print data
        if data:
            try:
                dataTrust = data[expandUrlNameStr]['0']
                valueTrust = dataTrust[0]
                confTrust = dataTrust[1]
                value = valueTrust * confTrust / 100
                #print value[0]
                return value
            except KeyError:
                return 0
            
    def numMediaContent(tweets_data):
        userName = map(lambda tweet: tweet['user']['screen_name'] if tweet['user'] != None else None, tweets_data)
        for i in userName:
            userNameStr = str(i).encode('utf-8', 'ignore')
        #userNameStr = ''.join(str(i) for i in userName)
        #print userNameStr
        url = "https://twitter.com/"+userNameStr
        #print url
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        #tweets = soup.findAll('li',{"class":'js-stream-item'})
        try:
            for e in soup.findAll('div', {'class' : 'PhotoRail'}):
                photoid = e.find('span', {'class':'PhotoRail-headingText'}).text.strip()
                photoids = str(photoid)
            numMedia = [float(s) for s in photoid.split() if all(j.isdigit() or j in string.punctuation for j in s) and any(j.isdigit() for j in s)]
            for i in numMedia:
                numMediaValue = i
            return i
        except:
            return 0

    def hasProfileImg(tweets_data):
        profileImg = map(lambda tweet: tweet['user']['default_profile_image'], tweets_data)
        for i in profileImg:
            profileImgStr = str(i).encode('utf-8', 'ignore')
        return profileImgStr

    def hasHeaderImg(tweets_data):
        headerImg = map(lambda tweet: tweet['user']['profile_use_background_image'], tweets_data)
        #headerImgStr = ''.join(str(i) for i in headerImg)
        for i in headerImg:
            headerImgStr = str(i).encode('utf-8', 'ignore')
        return headerImgStr

    def getAccountAge(tweets_data):
        age = map(lambda tweet: tweet['user']['created_at'], tweets_data)
        for i in age:
            ageStr = i.encode('utf-8', 'ignore')
        #age = ''.join(str(i) for i in age)
        #print age
        age1 = parse(ageStr)
        age1 = str(age1)
        #print age1
        try:
            mytime = datetime.strptime(age1, "%Y-%m-%d %H:%M:%S").strftime('%Y-%m-%d %H:%M:%S')
        except ValueError as v:
            if len(v.args) > 0 and v.args[0].startswith('unconverted data remains: '):
                age1 = age1[:-(len(v.args[0]) - 26)]
                mytime = datetime.strptime(age1, "%Y-%m-%d %H:%M:%S").strftime('%Y-%m-%d %H:%M:%S')
        # start of epoch time
        epoch = dt.datetime.utcfromtimestamp(0) 
        # plugin your time object
        my_time = dt.datetime.strptime(mytime, "%Y-%m-%d %H:%M:%S") 
        delta = my_time - epoch
        return delta.total_seconds()

    def getTweetRatio(tweets_data, accountAge, numTweets):
        timeCreated = map(lambda tweet: tweet['created_at'], tweets_data)
        for i in timeCreated:
            timeCreatedStr = i.encode('utf-8', 'ignore')
        #timeCreatedStr = ''.join(str(i) for i in timeCreated)
        #print timeCreatedStr
        timeCreatedStr1 = parse(timeCreatedStr)
        timeCreatedStr1 = str(timeCreatedStr1)

        try:
            mytime = datetime.strptime(timeCreatedStr1, "%Y-%m-%d %H:%M:%S").strftime('%Y-%m-%d %H:%M:%S')
        except ValueError as v:
            if len(v.args) > 0 and v.args[0].startswith('unconverted data remains: '):
                timeCreatedStr1 = timeCreatedStr1[:-(len(v.args[0]) - 26)]
                mytime = datetime.strptime(timeCreatedStr1, "%Y-%m-%d %H:%M:%S").strftime('%Y-%m-%d %H:%M:%S')

        # start of epoch time
        epoch = dt.datetime.utcfromtimestamp(0) 
        # plugin your time object
        my_time = dt.datetime.strptime(mytime, "%Y-%m-%d %H:%M:%S") 
        delta = my_time - epoch
        timeCreatedValue = delta.total_seconds()

        if accountAge != None:
            tweetRatio = (float(numTweets) / float(timeCreatedValue - accountAge) * 86400L)
            tweetRatio = str(tweetRatio)
            return tweetRatio
        else:
            return 0
        
    def getIndegree(tweetsara,externLink, file_name):
	if externLink == None:
		return 0
  	with open(file_name, 'rb') as tsvin:
		tsvreader = csv.reader(tsvin,delimiter="\t")
		for row in tsvreader:
			if expandedLink == row[0]:
				return row[1]
				#break
    
    
    print "getting tweet ids", time.time() - start_time
    tweet_id.append(getTweetId(tweets_data)) 
    numFriends.append(getNumFriends(tweets_data))
    numFollowers.append(getNumFollowers(tweets_data))
    numfriends = getNumFriends(tweets_data)
    numfollowers = getNumFollowers(tweets_data)
    timesListed.append(getTimesListed(tweets_data))
    followerFriendRatio.append(getFollowerFriendRatio(numfollowers, numfriends))
    hasUrlCheck.append(hasUrl(tweets_data))
    userUrl.append(getUserUrl(tweets_data))
    bioCheck.append(hasBio(tweets_data))
    verifiedUser.append(isVerifiedUser(tweets_data))
    numTweets = getNumTweets(tweets_data)

    numFavorites.append(getNumFavorites(tweets_data))
    locationCheck.append(hasLocation(tweets_data))
    existingLocationCheck.append(hasExistingLocation(tweets_data))
    externLink = checkForExternalLinks(tweets_data)
    #print externLink
    profileImgCheck.append(hasProfileImg(tweets_data))
    headerImgCheck.append(hasHeaderImg(tweets_data))
    accountAge.append(getAccountAge(tweets_data))
    accountage = getAccountAge(tweets_data)
    #print accountAge
    tweetRatio.append(getTweetRatio(tweets_data, accountage, numTweets))
    mediaContent.append(numMediaContent(tweets_data))
    
    print "getting Indegree and harmonic", time.time() - start_time
    externalLink = checkForExternalLinks(tweets_data)
    expandedLink = expandedUrl(externalLink)
    #print expandedLink
    if expandedLink == "twitter.com":
        expandedLink = None
    indegreeval = 0
    indegreeval = getIndegree(tweets_data,expandedLink,"D:/Downloads/hostgraph-indegree.tsv/hostgraph-indegree.tsv")
    #print indegreeval
    harmonicval = 0
    if indegreeval != 0:
         harmonicval = getIndegree(tweets_data, expandedLink, "D:/Downloads/hostgraph-h.tsv/hostgraph-h.tsv")
    Indegree.append(indegreeval)
    #Indegree.append(getIndegree(externLink, "D:/Downloads/hostgraph-indegree.tsv/hostgraph-indegree.tsv"))
    Harmonic.append(harmonicval)
    #Harmonic.append(getIndegree(externLink, "D:/Downloads/hostgraph-h.tsv/hostgraph-h.tsv"))
    print "got Indegree and harmonic", time.time() - start_time
    AlexaPopularity.append(getAlexaPopularity(expandedLink))
    AlexaReach.append(getAlexaReachRank(expandedLink))
    AlexaDelta.append(getAlexaDeltaRank(expandedLink))
    AlexaCountry.append(getAlexaCountryRank(expandedLink))
    WotValue.append(getWotTrustValue(externLink))
    

print "got all values in", time.time() - start_time    
    
for i in range(0,len(tweets_data)):
    ufl = [tweet_id[i], numFriends[i], numFollowers[i], followerFriendRatio[i], timesListed[i], hasUrlCheck[i], verifiedUser[i], 
           bioCheck[i], locationCheck[i], existingLocationCheck[i], WotValue[i],mediaContent[i], accountAge[i], profileImgCheck[i], headerImgCheck[i], 
           tweetRatio[i], Indegree[i], Harmonic[i], AlexaCountry[i], AlexaDelta[i], AlexaPopularity[i], AlexaReach[i]]
    ufl_result.append(ufl)
    
user_df = pd.DataFrame(list(ufl_result))
#item_df
pd.set_option('display.max_columns', None)
"""user_df.columns = ['tweetid', 'numFriends', 'numFollowers', 'followerFriendRatio', 'timesListed', 'hasUrlCheck', 'verifiedUser','bioCheck',
                   'locationCheck', 'existingLocationCheck', 'wotTrustValue', 'mediaContent', 'accountAge', 'profileImgCheck', 'headerImgCheck', 
                   'tweetRatio', 'Indegree', 'Harmonic','alexaCountryRank', 'alexaDeltaRank', 'alexaPopularity', 'alexaReachRank']
"""
user_df.to_pickle('trainingUserFeatures.pickle')
df3 = pd.read_pickle('trainingUserFeatures.pickle')
df3.columns = ['tweetid', 'numFriends', 'numFollowers', 'followerFriendRatio', 'timesListed', 'hasUrlCheck', 'verifiedUser','bioCheck',
                   'locationCheck', 'existingLocationCheck', 'wotTrustValue', 'mediaContent', 'accountAge', 'profileImgCheck', 'headerImgCheck', 
                   'tweetRatio', 'Indegree', 'Harmonic','alexaCountryRank', 'alexaDeltaRank', 'alexaPopularity', 'alexaReachRank']

df3[['Indegree', 'Harmonic']] = df3[['Indegree','Harmonic']].apply(pd.to_numeric)
df3['Indegree'].fillna(value='0', inplace = True)
df3['Harmonic'].fillna(value='0', inplace = True)
df3.to_pickle('trainingUserFeatures.pickle')
dfset = pd.read_pickle('trainingUserFeatures.pickle')
#print "My program took", time.time() - start_time, "to run"
#print dfset

