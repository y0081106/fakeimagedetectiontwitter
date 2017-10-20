import numpy as np
import json
import re
import requests
import string
import urllib, json
import urllib2
import time
import nltk
import csv
import sys
import pickle
import pandas as pd

from bs4 import BeautifulSoup
from PIL import Image
from textstat.textstat import textstat
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from langdetect import detect
from collections import Counter


F_HARMONIC = "D:/Downloads/hostgraph-h.tsv/hostgraph-h.tsv"
F_INDEGREE = "D:/Downloads/hostgraph-indegree.tsv/hostgraph-indegree.tsv"

F_PLEASE = "C:/Users/imaad/twitteradvancedsearch/senti_words/please.txt"
F_HAPPYEMO = "C:/Users/imaad/twitteradvancedsearch/emoticons/happy-emoticons.txt"
F_SADEMO = "C:/Users/imaad/twitteradvancedsearch/emoticons/sad-emoticons.txt"
F_FIRSTPRON = "C:/Users/imaad/twitteradvancedsearch/pronouns/first-order-prons.txt"
F_SECONDPRON = "C:/Users/imaad/twitteradvancedsearch/pronouns/second-order-prons.txt"
F_THIRDPRON = "C:/Users/imaad/twitteradvancedsearch/pronouns/third-order-prons.txt"
F_SLANG = "C:/Users/imaad/twitteradvancedsearch/slang_words/slangwords.txt"
F_NEGATIVE = "C:/Users/imaad/twitteradvancedsearch/senti_words/negative-words.txt"
F_POSITIVE = "C:/Users/imaad/twitteradvancedsearch/senti_words/positive-words.txt"

def getNumUppercaseChars(tweet):
    count = 0
    postText = tweet['text']
    for i in postText:
        postTextStr = i.encode('utf-8', 'ignore')
    postTextStr =  re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', postTextStr) # URLs
    postTextStr =  re.sub(r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)','', postTextStr) # remove hash-tags
    postTextStr =  re.sub(r'(?:@[\w_]+)','', postTextStr) # remove @-mentions
    for i in postTextStr:
        if i.isupper():
            count = count+1
    return count


def getNumUrls(tweets_data):
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

def get_alexa_metrics(domain):
    metrics = (0, 0, 0 ,0)
    if domain == None:
        return metrics
    url = "http://data.alexa.com/data?cli=10&url="+ domain
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    try:
        popularity = soup.popularity['text']
        rank = soup.reach['rank']
        country = soup.country['rank']
        delta = soup.rank['delta']
        return (popularity, rank, country, delta)
    except TypeError:
        return metrics

def hasExternalLinks(tweets_data):
        extLinks = checkForExternalLinks(tweets_data)
        if extLinks:
            return True
        else:
            return False

def getWotTrustValue(expandUrlName):
    if "/" in expandUrlName:
        expandUrlName1 = expandUrlName.split("/")[2:3]
    else:
        expandUrlName1 = expandUrlName.split("/")[:]
    expandUrlNameStr = str(expandUrlName1[0])
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

def numNouns(postText):
    is_noun = lambda pos: pos[:2] == 'NN'
    for i in postText:
        postTextStr = i.encode('utf-8', 'ignore')
    #postTextStr = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",postTextStr).split())
    #print postTextStr
    for i in postTextStr:
        tokenized = nltk.word_tokenize(postTextStr)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
    countNouns = Counter([j for i,j in pos_tag(word_tokenize(postTextStr))])
    return countNouns['NN']

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
    except AttributeError, IOError:
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
        #print e
        expandedUrlNameStr = None
    return expandedUrlNameStr

def checkForExternalLinks(tweet):
    numurls1 = []
    urlList = []
    numUrls1 = tweet['entities']['urls'] 
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

def getIndegree(expandedLink, fin=F_INDEGREE):
    #print expandedLink
    if expandedLink == "twitter.com":
        expandedLink = None
    indegreeval = None

    if expandedLink == None:
    	return 0
    with open(fin, 'rb') as tsvin:
    	tsvreader = csv.reader(tsvin,delimiter="\t")
    	for row in tsvreader:
    		if expandedLink == row[0]:
    			return row[1]

def getHarmonic(indegree, expandedLink, f_harmonic):
    if indegree == None:
        return None
    return getIndegree(expandedLink, F_HARMONIC)


def read_pattern(fin, regex=False):
    with open(fin) as f:
        patterns = f.readlines()
    return [re.compile(r'\b%s\b' % p.strip()) for p in patterns]


def pattern_count(ttext, pattern):
    count = 0
    for p in pattern:
        matches = p.findall(ttext)
        count += len(matches)
    return count > 0, count

PATTERN_PLEASE = read_pattern(F_PLEASE)
PATTERN_HAPPYEMO = read_pattern(F_HAPPYEMO)
PATTERN_SADEMO  = read_pattern(F_SADEMO)
PATTERN_FIRSTPRON = read_pattern(F_FIRSTPRON)
PATTERN_SECPRON = read_pattern(F_SECONDPRON)
PATTERN_THIRDPRON = read_pattern(F_THIRDPRON)
PATTERN_SLANG = read_pattern(F_SLANG)
PATTERN_NEGATIVE = read_pattern(F_NEGATIVE)
PATTERN_POSITIVE = read_pattern(F_POSITIVE)

header = ('id',
          'tweetTextLen',
          'numItemWords',
          'questionSymbol',
          'exclamSymbol',
          'numQuesSymbol',
          'numExclamSymbol',
          'happyEmo',
          'sadEmo',
          'numUpperCase',
          'containFirstPron',
          'containSecPron',
          'containThirdPron',
          'numMentions',
          'numHashtags',
          'numUrls',
          'positiveWords',
          'negativeWords',
          'slangWords',
          'pleasePresent',
          'rtCount',
          'colonSymbol',
          'externLinkPresent',
          'Indegree',
          'Harmonic',
          'AlexaPopularity',
          'AlexaReach',
          'AlexaDelta',
          'AlexaCountry',
          'WotValue',
          'numberNouns',
          'readabilityValue')

def gen_features(tweet):
	print "Hello"
	tid = tweet['id_str']
	ttext = tweet['text']
	tlength = len(ttext)
	twords = len(ttext.split())
	counts = Counter(ttext)
	questionSymbol = '?' in counts
	exclamSymbol = '!' in counts
	numQuesSymbol = counts.get('?', 0)
	numExclamSymbol = counts.get('!', 0)
	numUpperCase = getNumUppercaseChars(tweet)
	numMentions = len(tweet['entities'].get('user_mentions', []))
	numHashtags = len(tweet['entities'].get('user_hashtags', []))
	numUrls = len(tweet['entities'].get('urls', [])) + len(tweet['entities'].get('media', []))
	rtCount = tweet.get('retweet_count', 0)
	colonSymbol = ':' in counts
	externLinkPresent = len(tweet['entities'].get('urls', [])) > 0
	externalLink = checkForExternalLinks(tweet)
	expandedLink = expandedUrl(externalLink)
	indegree = getIndegree(expandedLink)
	harmonic = getHarmonic(indegree, expandedLink, F_HARMONIC)
	alexa_metrics = get_alexa_metrics(expandedLink)
	wotValue = getWotTrustValue(externalLink)
	numberNouns = numNouns(ttext)
	readabilityValue = textstat.flesch_reading_ease(str(' '.join(ttext).encode('utf-8').strip()))
	please_exists, _ = pattern_count(ttext, PATTERN_PLEASE)
	containsFirstPron, _ = pattern_count(ttext, PATTERN_FIRSTPRON)
	containsSecPron, _ = pattern_count(ttext, PATTERN_SECPRON)
	containsThirdPron, _ = pattern_count(ttext, PATTERN_THIRDPRON)
	containsHappyEmo, _ = pattern_count(ttext, PATTERN_HAPPYEMO)
	containsSadEmo, _ = pattern_count(ttext, PATTERN_SADEMO)
	_, slangWords = pattern_count(ttext, PATTERN_SLANG)
	_, negWords = pattern_count(ttext, PATTERN_NEGATIVE)
	_, posWords = pattern_count(ttext, PATTERN_POSITIVE)
	features = (tid,
                tlength,
                twords,
                questionSymbol,
                exclamSymbol,
				externLinkPresent,
				numberNouns,
				containsHappyEmo,
				containsSadEmo,
				containsFirstPron,
				containsSecPron,
				containsThirdPron,
				numUpperCase,
				posWords,
				negWords,
				numMentions,
                numHashtags,
				numUrls,
				rtCount,
				slangWords,
				colonSymbol,
				please_exists,
				wotValue,
                numQuesSymbol,
                numExclamSymbol,
				readabilityValue,
                indegree,
                harmonic,
                alexa_metrics)
	return features



def read_args():
    if len(sys.argv) == 2:
        tweets_data_path = sys.argv[1]
    else:
        tweets_data_path = 'C:/Users/imaad/twitteradvancedsearch/real_tweets.json'
	return tweets_data_path
def main():
	fin = read_args()
	with open(fin) as f:
		for line in f:
			tweet = json.loads(line)
			tweet_features = gen_features(tweet)
			print tweet_features

if __name__ == '__main__':
    main()