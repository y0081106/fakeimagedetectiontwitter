import pandas as pd
import numpy as np
import json
import re
import emoji
import requests
import string
import urllib, json
import nltk
import urllib2
import time
from bs4 import BeautifulSoup
from PIL import Image
from textstat.textstat import textstat
from langdetect import detect

#opening the json file with tweets
tweets_data = []
tweets_data_path = 'C:/Users/imaad/twitteradvancedsearch/real_tweets.json'
tweets_file = open(tweets_data_path, "r")

#loading the json file line by line
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue 
		
    #function to get the tweet id
    def getTweetId(tweets_data):
        twId = map(lambda tweet: tweet['id_str'], tweets_data)
		#encoding to utf-8
        for i in twId:
            twIdStr = i.encode('utf-8', 'ignore')
        return twIdStr
		
	#function to get the length of tweet's text
    def getText(tweets_data):
        postText = map(lambda tweet: tweet['text'], tweets_data)
		#encoding to utf-8
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
        return len(postTextStr)
	
	#function to get the number of words in a tweet's text
    def getNumItemWords(tweets_data):
        postText = map(lambda tweet: tweet['text'], tweets_data)
        #encoding to utf-8
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
		#counting the number of words
        postTextStrLen= len(postTextStr.split())
        return postTextStrLen

	#function to check if a given symbol is present 
    def containsSymbol(tweets_data, symbol):
        postText = map(lambda tweet: tweet['text'], tweets_data)
        #encoding to utf-8
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
        #remove url
        postTextStr =  re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', postTextStr)
        #looping through to find symbol
		for i in postTextStr.strip():
            if i == symbol:
                return True
        return False

	#function to count the number of occurrence of given symbol	
    def getNumSymbol(tweets_data, symbol):
        count = 0
        postText = map(lambda tweet: tweet['text'], tweets_data)
        #encoding to utf-8
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
		#remove url
        postTextStr =  re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', postTextStr)
        #looping through and increasing count if symbol is found
		for i in postTextStr.strip():
            if i == symbol:
                count = count + 1
        return count

	#function to check for emotion of the text	
    def containsEmo(tweets_data, file_name):
        postText = map(lambda tweet: tweet['text'], tweets_data)
        #encoding to utf-8
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
		#open the file which contains words corresponding to the emotion
        textfile = open(file_name, 'r')
        filetext = textfile.readlines()
        textfile.close()
		#compare emotional words to text
        for i in filetext:
            if re.findall(i, postTextStr):
                return True
            else:
                return False

	#function to get number of uppercase characters
    def getNumUppercaseChars(tweets_data):
        count = 0
        postText = map(lambda tweet: tweet['text'], tweets_data)
        #encoding to utf-8
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
		#remove URLs
        postTextStr =  re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', postTextStr) 
		#remove hashtags
        postTextStr =  re.sub(r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)','', postTextStr)
		#remove @-mentions
        postTextStr =  re.sub(r'(?:@[\w_]+)','', postTextStr) 
        #loop through to count uppercase characters
        for i in postTextStr:
            if i.isupper():
                count = count+1
        return count

	#function to check if text contains pronoun
    def containsPron(tweets_data, file_name):
        postText = map(lambda tweet: tweet['text'], tweets_data)
        #encoding to utf-8
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
		#convert string to lowercase
        postTextStr = str.lower(postTextStr)
		#open the file which contains words corresponding to pronouns
        textfile = open(file_name, 'r')
        filetext = textfile.read().splitlines()
        textfile.close()
		#looping thorugh to search for pronouns
        for i in filetext:
             if re.search(i, postTextStr):
                print i
                return True
             else:
                continue
        return False

	#function to get number of mentions	
    def getNumMentions(tweets_data):
        numMentions = map(lambda tweet: tweet['entities']['user_mentions'] if tweet['entities'] != None else None, tweets_data)
        mentions = []
		#if mentions are present in tweet, store them
        if len(numMentions) > 0:
            mentions.extend(tweet['entities']['user_mentions'])
            tweet['mention_names'] = [mention['screen_name'] for mention in mentions]
		#return the length of number of mentions
        return len(tweet['mention_names'])
	
	#function to get number of hashtags
    def getNumHashtags(tweets_data):
        numHashtags = map(lambda tweet: tweet['entities']['hashtags'] if tweet['entities'] != None else None, tweets_data)
        hashtags = []
		#if hashtags are present in tweet, store them.
        if len(numHashtags) > 0:
            hashtags.extend(tweet['entities']['hashtags'])
            tweet['hashtags'] = [tag['text'] for tag in hashtags]
        #return the length of number of hashtags
        return len(tweet['hashtags'])
	
	#function to get the number of URLs
    def getNumUrls(tweets_data):
        numUrls1 = map(lambda tweet: tweet['entities']['urls'] if tweet['entities'] != None else None, tweets_data)
        numurls1 = []
		#if urls present in the tweet, proceed
        if len(numUrls1) > 0:
            numurls1.extend(tweet['entities']['urls'])
            tweet['url1'] = [numurl['url'] for numurl in numurls1]
        #check for media urls
        if 'media' in tweet['entities']:
            numUrls2 = map(lambda tweet: tweet['entities']['media'] if tweet['entities'] != None else None, tweets_data)
            numurls2 = []
			#if media urls present in the tweet, proceed
            if len(numUrls2) > 0:
                numurls2.extend(tweet['entities']['media'])
                tweet['url2'] = [numurl['url'] for numurl in numurls2]
			#sum the total number of urls
            totalurl = len(tweet['url1']) + len(tweet['url2'])
        else:
            totalurl = len(tweet['url1'])
        return totalurl
	
	#function to get number of sentimental words in text of the tweet
    def getNumSentiWords(tweets_data, file_name):
        count = 0
        postText = map(lambda tweet: tweet['text'], tweets_data)
        #encoding to utf-8
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
		#convert text to lowercase
        postTextStr = str.lower(postTextStr)
        #opening the file with the sentimental words
        textfile = open(file_name, 'r')
        filetext = textfile.read().splitlines()
        textfile.close()
		#looping thorugh to search for sentimental words
        for i in filetext:
            k = re.compile(r'\b%s\b' % i)
            if k.search(postTextStr):
                count = count + 1
            else:
                continue
        if (count >=1):
            return count
        else:
            return 0
			
	#function to check if tweet's text contains 'please'
    def hasPlease(tweets_data, file_name):
        count = 0
        postText = map(lambda tweet: tweet['text'], tweets_data)
        #encoding to utf-8
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
		#converting the text to lowercase
        postTextStr = str.lower(postTextStr)
        textfile = open(file_name, 'r')
        filetext = textfile.read().splitlines()
        textfile.close()
        for i in filetext:
			#compiling each word in the file
            k = re.compile(r'\b%s\b' % i)
			#searching for the word in the text of the tweet
            if k.search(postTextStr):
				#if word is found, increment count by 1
                count = count + 1
            else:
                continue
		#if word occurs, then return True else return False
        if (count >=1):
            return True
        else:
            return False
			
	#function to get the retweets count
    def getRetweetsCount(tweets_data):
		#if retweet field present, get it's value
        if 'retweet_count' in tweet:
            numRetweets= map(lambda tweet: tweet['retweet_count'], tweets_data)
			#encoding to utf-8
            for i in numRetweets:
                numRetweetsStr = str(i).encode('utf-8', 'ignore')
            return numRetweetsStr
        else:
            return 0

	#function to get number of slang words in the text of the tweet
    def getNumSlangWords(tweets_data, file_name):
        count = 0
        postText = map(lambda tweet: tweet['text'], tweets_data)
        #encoding to utf-8
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
        postTextStr = str.lower(postTextStr)
        #opening the file with the slang words
        textfile = open(file_name, 'r')
        filetext = textfile.read().splitlines()
        textfile.close()
        for i in filetext:
			#compiling the slang words as a pattern 
            k = re.compile(r'\b%s\b' % i)
			#search for the slang word in the text string
            if k.search(postTextStr):
				#increment count when there is a match
                count = count + 1
            else:
                continue
		#if there is/are slang word/s, return their count
        if (count >=1):
            return count
        else:
            return 0
			
	#function to get Alexa Popularity of a URL
    def getAlexaPopularity(domain):
		#if URL is not present, return 0
        if domain == None:
            return 0
		#url to access the API of alexa.com
        url = "http://data.alexa.com/data?cli=10&url="+ domain
		#get the response
        response = requests.get(url)
		#parsing the response and return popularity 
        soup = BeautifulSoup(response.text, 'html.parser')
        try:
            alexaPopularity = soup.popularity['text']
            return alexaPopularity
        except TypeError:
            return 0
	
	#function to get Alexa Reach Rank of a URL
    def getAlexaReachRank(domain):
	#if URL is not present, return 0
        if domain == None:
            return 0
		#url to access the API of alexa.com
        url = "http://data.alexa.com/data?cli=10&url="+ domain
		#get the response
        response = requests.get(url)
		#parsing the response and return reach rank
        soup = BeautifulSoup(response.text, 'html.parser')
        try:
            alexaReachRank = soup.reach['rank']
            return alexaReachRank
        except TypeError:
            return 0
			
	#function to get Alexa Delta Rank of a URL
    def getAlexaDeltaRank(domain):
	#if URL is not present, return 0
        if domain == None:
            return 0
		#url to access the API of alexa.com	
        url = "http://data.alexa.com/data?cli=10&url="+ domain
		#get the response
        response = requests.get(url)
		#parsing the response and return delta rank
        soup = BeautifulSoup(response.text, 'html.parser')
        try:
            alexaDeltaRank = soup.rank['delta']
            return alexaDeltaRank
        except TypeError:
            return 0
	
	#function to get Alexa Country Rank of a URL
    def getAlexaCountryRank(domain):
	#if URL is not present, return 0
        if domain == None:
            return 0
		#url to access the API of alexa.com	
        url = "http://data.alexa.com/data?cli=10&url="+ domain
		#get the response
        response = requests.get(url)
		#parsing the response and return country rank
        soup = BeautifulSoup(response.text, 'html.parser')
        try:
            alexaCountryRank = soup.country['rank']
            return alexaCountryRank
        except TypeError:
            return 0
			
	#function to check if url is an image
    def isAnImage(url):
		#url pattern of images stored on twitter
        text = 'pbs.twimg.com'
        text1 = 'p.twimg.com'
		#if the pattern matches, then image url is present
        if re.search(url, text) or re.search(url, text1):
            return True
		#opening the image
        try:
            im = Image.open(urllib2.urlopen(url))
			#if image can be opened, return True
            if im != None:
                return True
            else:
                return False
		#error handling
        except IOError:
            return False
			
	#function to check for the presence of external links
    def checkForExternalLinks(tweets_data):
        numurls1 = []
        urlList = []
        numUrls1 = map(lambda tweet: tweet['entities']['urls'] if tweet['entities'] != None else None, tweets_data)
        if len(numUrls1) > 0:
            numurls1.extend(tweet['entities']['urls'])
            tweet['url1'] = [numurl['url'] for numurl in numurls1]
            urlList.extend(tweet['url1'])
        #check if external link is an image
        for ural in urlList:
            checkForImage = isAnImage(ural)
            if  not checkForImage:
                return ural
            else:
                return None
				
	#function to check if there are external links
    def hasExternalLinks(tweets_data):
            extLinks = checkForExternalLinks(tweets_data)
            if extLinks:
                return True
            else:
                return False

	#function to expand URL
    def expandUrl(shortenedUrl):
        expandedUrl = (requests.get(shortenedUrl).url)
        expandedUrlName = expandedUrl.split("/")[2:3]
        expandedUrlNameStr = str(expandedUrlName[0])
        return expandedUrlNameStr

	#function to get the WOT trust value
    def getWotTrustValue(host):
	#if the link is none, return 0
        if host == None:
            return 0
		#expand the URL
        expandUrlName = expandUrl(host)
        if "/" in expandUrlName:
            expandUrlName1 = expandUrlName.split("/")[2:3]
        else:
            expandUrlName1 = expandUrlName.split("/")[:]
        expandUrlNameStr = str(expandUrlName1[0])
        value = [None]
		#accessing the mywot api
        url = "http://api.mywot.com/0.4/public_link_json2?hosts="+ expandUrlNameStr +"/&key=108d4b2a42ea1afc370e668b39cabdceaa19fcf0"
        #get the response
        response = urllib.urlopen(url)
        data = json.load(response)
        #if values are available, then calculate the value of WOT 
        if data:
            try:
                dataTrust = data[expandUrlNameStr]['0']
                valueTrust = dataTrust[0]
                confTrust = dataTrust[1]
                value = valueTrust * confTrust / 100
                return value
		#error handling
            except KeyError:
                return 0
	
	#function to count the number of nouns
    def numNouns(tweets_data):
        postText = map(lambda tweet: tweet['text'], tweets_data)
        is_noun = lambda pos: pos[:2] == 'NN' 
        #encoding to utf-8
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
        postTextStr = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",postTextStr).split())
        for i in postTextStr:
            tokenized = nltk.word_tokenize(postTextStr)
            nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
        countNouns = Counter([j for i,j in pos_tag(word_tokenize(postTextStr))])
        return countNouns['NN']
	
	#function to get readability
    def getReadability(tweets_data):
        postText = map(lambda tweet: tweet['text'], tweets_data)
        #encoding to utf-8
        for i in postText:
            postTextStr = i.encode('utf-8', 'ignore')
        readability = textstat.flesch_reading_ease(str(postTextStr))
        return readability
    
    tweetid = str(getTweetId(tweets_data))
    tweetText = str(getText(tweets_data))
    numItemWords = str(getNumItemWords(tweets_data))
    questionSymbol = str(containsSymbol(tweets_data, '?')) 
    exclamSymbol = str(containsSymbol(tweets_data, '!'))
    numQuesSymbol = str(getNumSymbol(tweets_data, '?'))
    numExclamSymbol = str(getNumSymbol(tweets_data, '!'))
    happyEmo = str(containsEmo(tweets_data, 'C:/Users/imaad/twitteradvancedsearch/emoticons/happy-emoticons.txt'))
    sadEmo =  str(containsEmo(tweets_data, 'C:/Users/imaad/twitteradvancedsearch/emoticons/sad-emoticons.txt'))
    numUpperCase = str(getNumUppercaseChars(tweets_data))
    containPron = str(containsPron(tweets_data, 'C:/Users/imaad/twitteradvancedsearch/pronouns/first-order-prons.txt'))
    numMentions = str(getNumMentions(tweets_data))
    numHashtags = str(getNumHashtags(tweets_data))
    numUrls = str(getNumUrls(tweets_data))
    positiveWords = str(getNumSentiWords(tweets_data, 'C:/Users/imaad/twitteradvancedsearch/senti_words/positive-words.txt' ))
    negativeWords = str(getNumSentiWords(tweets_data, 'C:/Users/imaad/twitteradvancedsearch/senti_words/negative-words.txt' ))
    slangWords = str(getNumSlangWords(tweets_data, 'C:/Users/imaad/twitteradvancedsearch/slang_words/slangwords.txt' ))
    pleasePresent = str(hasPlease(tweets_data,'C:/Users/imaad/twitteradvancedsearch/senti_words/please.txt'))
    rtCount = str(getRetweetsCount(tweets_data))
    colonSymbol = str(containsSymbol(tweets_data, ':'))
    externLinkPresent = str(hasExternalLinks(tweets_data))
    externLink = checkForExternalLinks(tweets_data)
    AlexaPopularity = str(getAlexaPopularity(externLink))
    AlexaReach = str(getAlexaReachRank(externLink))
    AlexaDelta = str(getAlexaDeltaRank(externLink))
    AlexaCountry = str(getAlexaCountryRank(externLink))
    WotValue = str(getWotTrustValue(externLink))
    numberNouns = str(numNouns(tweets_data))    
    readabilityValue = str(getReadability(tweets_data))
	#storing all features in a list
    ifl = [tweetid, tweetText, numItemWords, questionSymbol, exclamSymbol, numQuesSymbol, numExclamSymbol, happyEmo,
          sadEmo, numUpperCase, containPron, numMentions, numHashtags, numUrls, positiveWords, negativeWords,
          slangWords,pleasePresent, rtCount, colonSymbol, externLinkPresent, AlexaPopularity, AlexaReach, AlexaDelta,
          AlexaCountry, WotValue, numberNouns, readabilityValue]
    %store (str(ifl)) >> real_tweets.txt
