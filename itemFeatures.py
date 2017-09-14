import numpy as np
import json
import re
import requests
import string
import urllib, json
import time
import nltk
import csv
import pickle
import pandas as pd

from bs4 import BeautifulSoup
from PIL import Image
from textstat.textstat import textstat
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from langdetect import detect
from collections import Counter

start_time = time.time()
tweets_data_path = 'C:/Users/imaad/twitteradvancedsearch/papertweetstrial.json'
tweets_data, ifl_result, tweet_id, tweetText, tweetTextLen, numItemWords, questionSymbol, exclamSymbol,
numQuesSymbol, numExclamSymbol, happyEmo,sadEmo, numUpperCase, containFirstPron,containSecPron,containThirdPron,
numMentions, numHashtags, numUrls, positiveWords, negativeWords, slangWords, pleasePresent, rtCount,  
colonSymbol, externLinkPresent, indegreeval, harmonicval, AlexaPopularity, AlexaReach, AlexaDelta, AlexaCountry,
WotValue, numberNouns, readabilityValue, Indegree, Harmonic  = []   ([] for i in range(37))

tweets_file = open(tweets_data_path, "r")          
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue     
    tweets = pd.DataFrame()
    
    def getTweetId(tweets_data):
        twId = map(lambda tweet: tweet['id_str'], tweets_data)
        for i in twId:
            twIdStr = i.encode('utf-8', 'ignore')
        #postTextStr = ''.join(str(i) for i in postText)
        return twIdStr

    def getText(tweets_data):
        postText = map(lambda tweet: tweet['full_text'], tweets_data)
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
        #postTextStr = ''.join(str(i) for i in postText)
        return postTextStr
    
    def getTextLen(tweets_data):
        postText = map(lambda tweet: tweet['full_text'], tweets_data)
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
        #postTextStr = ''.join(str(i) for i in postText)
        return len(postTextStr)

    def getNumItemWords(tweets_data):
        postText = map(lambda tweet: tweet['full_text'], tweets_data )
        #postTextStr = ''.join(str(i) for i in postText)
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
        postTextStrLen= len(postTextStr.split())
        return postTextStrLen

    def containsSymbol(tweets_data, symbol):
        postText = map(lambda tweet: tweet['full_text'], tweets_data)
        #postTextStr = ''.join(str(i) for i in postText)
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
        #remove url
        postTextStr =  re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', postTextStr)
        for i in postTextStr.strip():
            if i == symbol:
                return True
        return False

    def getNumSymbol(tweets_data, symbol):
        count = 0
        postText = map(lambda tweet: tweet['full_text'], tweets_data)
        #postTextStr = ''.join(str(i) for i in postText)
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
        postTextStr =  re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', postTextStr)
        for i in postTextStr.strip():
            if i == symbol:
                count = count + 1
        return count

    def getNumUppercaseChars(tweets_data):
        count = 0
        postText = map(lambda tweet: tweet['full_text'], tweets_data)
        #postTextStr = ''.join(str(i) for i in postText)
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
        postTextStr =  re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', postTextStr) # URLs
        postTextStr =  re.sub(r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)','', postTextStr) # remove hash-tags
        postTextStr =  re.sub(r'(?:@[\w_]+)','', postTextStr) # remove @-mentions
        #print postTextStr
        for i in postTextStr:
            if i.isupper():
                count = count+1
        return count
		
    def getNumMentions(tweets_data):
        numMentions = map(lambda tweet: tweet['entities']['user_mentions'] if tweet['entities'] != None else None, tweets_data)
        mentions = []
        if len(numMentions) > 0:
            mentions.extend(tweet['entities']['user_mentions'])
            tweet['mention_names'] = [mention['screen_name'] for mention in mentions]
        return len(tweet['mention_names'])

    def getNumHashtags(tweets_data):
        numHashtags = map(lambda tweet: tweet['entities']['hashtags'] if tweet['entities'] != None else None, tweets_data)
        hashtags = []
        if len(numHashtags) > 0:
            hashtags.extend(tweet['entities']['hashtags'])
            tweet['hashtags'] = [tag['text'] for tag in hashtags]
        return len(tweet['hashtags'])

    def getNumUrls(tweets_data):
        #numUrls1 = map(lambda tweet: tweet['entities']['urls'] if tweet['entities'] != None else None, tweets_data)
        numUrls1 = [tweet.get('entities''urls','') for tweet in tweets_data]
      
        numurls1 = []
      
        if len(numUrls1) > 0:
            #numurls1.extend(tweet.get('entities''media',''))
            numurls1.extend(tweet['entities']['urls'])
            tweet['url1'] = [numurl['url'] for numurl in numurls1]
        if 'media' in tweet['entities']:
            #numUrls2 = map(lambda tweet: tweet['entities']['media'] if tweet['entities'] != None else None, tweets_data)
            numUrls2 = [tweet.get('entities''media','') for tweet in tweets_data]
            numurls2 = []
            if len(numUrls2) > 0:
                numurls2.extend(tweet['entities']['media'])
                tweet['url2'] = [numurl['url'] for numurl in numurls2]
            totalurl = len(tweet['url1']) + len(tweet['url2'])
        else:
            totalurl = len(tweet['url1'])
        return totalurl

    def getRetweetsCount(tweets_data):
        if 'retweet_count' in tweet:
            numRetweets= map(lambda tweet: tweet['retweet_count'], tweets_data)
            for i in numRetweets:
                numRetweetsStr = str(i).encode('utf-8', 'ignore')
            return numRetweetsStr
        else:
            return 0

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

    def hasExternalLinks(tweets_data):
            extLinks = checkForExternalLinks(tweets_data)
            if extLinks:
                return True
            else:
                return False

    def getWotTrustValue(host):
        if host == None:
            return 0
        expandUrlName = expandedUrl(host)
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

    def numNouns(tweets_data):
        postText = map(lambda tweet: tweet['full_text'], tweets_data)
        is_noun = lambda pos: pos[:2] == 'NN' 
        #postTextStr = ''.join(str(i) for i in postText)
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
        postTextStr = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",postTextStr).split())
        #print postTextStr
        for i in postTextStr:
            tokenized = nltk.word_tokenize(postTextStr)
            nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
        countNouns = Counter([j for i,j in pos_tag(word_tokenize(postTextStr))])
        return countNouns['NN']

    def getReadability(tweets_data):
        postText = map(lambda tweet: tweet['full_text'], tweets_data)
        #postTextStr = ''.join(str(i) for i in postText)
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
        readability = textstat.flesch_reading_ease(str(postTextStr))
        return readability
    
        
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
            
    def expandedUrl(shortenedUrl):
        if shortenedUrl == None:
            return None
        try:
            expandedUrl = (requests.get(shortenedUrl).url)
            expandedUrlName = expandedUrl.split("/")[2:3]
            expandedUrlNameStr = str(expandedUrlName[0])
            expandedUrlNameStr = expandedUrlNameStr.replace("http://","")
            expandedUrlNameStr = expandedUrlNameStr.replace("www.", "")
        except requests.exceptions.ConnectionError as e:
            print e
            expandedUrlNameStr = None
        return expandedUrlNameStr
    
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
            
    def getIndegree(tweets_data,externLink, file_name):
	if externLink == None:
		return 0
  	with open(file_name, 'rb') as tsvin:
		tsvreader = csv.reader(tsvin,delimiter="\t")
		for row in tsvreader:
			if expandedLink == row[0]:
				return row[1]
				
    #print "getting tweet ids", time.time() - start_time
    tweet_id.append(getTweetId(tweets_data))
    #print "getting tweet text", time.time() - start_time
    tweetText.append(getText(tweets_data))
    tweetTextLen.append(getTextLen(tweets_data))
    numItemWords.append(getNumItemWords(tweets_data))
    questionSymbol.append(containsSymbol(tweets_data, '?'))
    exclamSymbol.append(containsSymbol(tweets_data, '!'))
    numQuesSymbol.append(getNumSymbol(tweets_data, '?'))
    numExclamSymbol.append(getNumSymbol(tweets_data, '!'))
    numUpperCase.append(getNumUppercaseChars(tweets_data))
    numMentions.append(getNumMentions(tweets_data))
    numHashtags.append(getNumHashtags(tweets_data))
    numUrls.append(getNumUrls(tweets_data))
    rtCount.append(getRetweetsCount(tweets_data))
    colonSymbol.append(containsSymbol(tweets_data, ':'))
    externLinkPresent.append(hasExternalLinks(tweets_data))
    externLink = checkForExternalLinks(tweets_data)
    #print "getting Indegree and harmonic", time.time() - start_time
    externalLink = checkForExternalLinks(tweets_data)
    expandedLink = expandedUrl(externalLink)
    #print expandedLink
    if expandedLink == "twitter.com":
        expandedLink = None
    indegreeval = None
    indegreeval = getIndegree(tweets_data,expandedLink,"D:/Downloads/hostgraph-indegree.tsv/hostgraph-indegree.tsv")
    #print indegreeval
    harmonicval = None
    if indegreeval != None:
         harmonicval = getIndegree(tweets_data, expandedLink, "D:/Downloads/hostgraph-h.tsv/hostgraph-h.tsv")
    Indegree.append(indegreeval)
    Harmonic.append(harmonicval)
    #print "got Indegree and harmonic", time.time() - start_time
    AlexaPopularity.append(getAlexaPopularity(externLink))
    AlexaReach.append(getAlexaReachRank(externLink))
    AlexaDelta.append(getAlexaDeltaRank(externLink))
    AlexaCountry.append(getAlexaCountryRank(externLink))
    WotValue.append(getWotTrustValue(externLink))
    numberNouns.append(numNouns(tweets_data))    
    readabilityValue.append(getReadability(tweets_data))

print "got all values in", time.time() - start_time    

def findPatternTrueFalse(file_name):
    values=[]
    with open(file_name,'r') as f:
        filetext = f.read().splitlines()
    for j in range(0, len(tweets_data)):
        count = False
        for i in filetext:
            p = re.compile(r'\b%s\b' % i)
            a = p.findall(tweetText[j])
        #if there is a match, i.e. 'a' has a value, then add the number of words matched to the count
            if a:
                count = True
        values.append(count)
    return values

def findPatternCount(file_name):
    values = []
    with open(file_name,'r') as f:
        filetext = f.read().splitlines()
    for j in range(0, len(tweets_data)):
        count = 0
        for i in filetext:
            p = re.compile(r'\b%s\b' % i)
            a = p.findall(tweetText[j])
        #if there is a match, i.e. 'a' has a value, then add the number of words matched to the count
            if a:
                count = count + len(a)
        values.append(count)
    return values

pleasePresent = findPatternTrueFalse("C:/Users/imaad/twitteradvancedsearch/senti_words/please.txt")
happyEmo = findPatternTrueFalse("C:/Users/imaad/twitteradvancedsearch/emoticons/happy-emoticons.txt")
sadEmo = findPatternTrueFalse("C:/Users/imaad/twitteradvancedsearch/emoticons/sad-emoticons.txt")
containFirstPron = findPatternTrueFalse("C:/Users/imaad/twitteradvancedsearch/pronouns/first-order-prons.txt")
containSecPron = findPatternTrueFalse("C:/Users/imaad/twitteradvancedsearch/pronouns/second-order-prons.txt")
containThirdPron = findPatternTrueFalse("C:/Users/imaad/twitteradvancedsearch/pronouns/third-order-prons.txt")
slangWords = findPatternCount("C:/Users/imaad/twitteradvancedsearch/slang_words/slangwords.txt")
negativeWords = findPatternCount("C:/Users/imaad/twitteradvancedsearch/senti_words/negative-words.txt")
positiveWords = findPatternCount("C:/Users/imaad/twitteradvancedsearch/senti_words/positive-words.txt")

for i in range(0,len(tweets_data)):
    ifl = [tweet_id[i], tweetTextLen[i], numItemWords[i], questionSymbol[i], exclamSymbol[i], numQuesSymbol[i], numExclamSymbol[i], happyEmo[i],
    sadEmo[i], numUpperCase[i], containFirstPron[i],containSecPron[i],containThirdPron[i], numMentions[i], numHashtags[i], numUrls[i], positiveWords[i], negativeWords[i],
    slangWords[i],pleasePresent[i], rtCount[i], colonSymbol[i], externLinkPresent[i],Indegree[i], Harmonic[i], AlexaPopularity[i], AlexaReach[i], AlexaDelta[i],
    AlexaCountry[i], WotValue[i], numberNouns[i], readabilityValue[i]]
    ifl_result.append(ifl)
   
file = open('paperTweetItemFeatures.txt', 'w')
for i in range(0, len(tweets_data)):
    file.write(str(ifl_result[i])+ '\n')
item_df = pd.DataFrame(list(ifl_result))
file.close()

pd.set_option('display.max_columns', None)
item_df.columns = ['id','tweetTextLen', 'numItemWords', 'questionSymbol', 'exclamSymbol', 'numQuesSymbol', 'numExclamSymbol', 'happyEmo',
          'sadEmo', 'numUpperCase', 'containFirstPron','containSecPron','containThirdPron', 'numMentions', 'numHashtags', 'numUrls', 'positiveWords', 'negativeWords',
          'slangWords','pleasePresent', 'rtCount', 'colonSymbol', 'externLinkPresent','Indegree','Harmonic', 'AlexaPopularity', 'AlexaReach', 'AlexaDelta',
          'AlexaCountry', 'WotValue', 'numberNouns', 'readabilityValue']
		  
item_df.to_pickle('TweetItemFeatures.pickle')
#df = pd.read_pickle('TweetItemFeatures.pickle')
