import numpy as np
import json
import re
import emoji
import requests
import string
import urllib, json
import urllib2
from bs4 import BeautifulSoup
from PIL import Image
from textstat.textstat import textstat
import datetime as dt
from datetime import datetime
from dateutil.parser import parse
from geopy.geocoders import Nominatim
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import time
from langdetect import detect
import nltk
from collections import Counter
import csv
import pickle
import pandas as pd

#Agreement Percentage
def agreementPercentage(agreedVal):
    a = []
    for i in range(len(agreedVal)):
        res = Counter(agreedVal[i])
        b = res.keys(), res.values()
        a.append(b)

    for i in range(len(a)):
        if len(a[i][0])==2:
            #printing the agreement accuracy if the there are 2 possibilies i.e disagreed and fake or real
            print float((a[i][1][1]))/sum(a[i][1])
        else:
            #printing the agreement accuracy if the there are 3 possibilies i.e disagreed and fake and real
            print float((a[i][1][1]+a[i][1][2]))/sum(a[i][1])
    return a
	
#Agreed Accuracy Caluclation
def agreedAccuracy(agreedVal):
    total_counter =[]
    counter = []
    acc = []
    for j in range(len(agreedVal)):
        count = 0
        total_count = 0
        for i in range(len(agreedVal[j])):
            if agreedVal[j][i] != 'disagreed':
                if agreedVal[j][i][0] == agreedVal[j][i][1]:
                    count= count +1
                    total_count = total_count + 1
                else:
                    total_count = total_count + 1
        counter.append(count)
        total_counter.append(total_count)

    #print final_count
    #print counter
    #print total_counter
    for i in range(len(counter)):
        acc.append(float(counter[i])/total_counter[i])
    return acc
	
#Function to calculate the accuracy of CL1 when the samples disagreed
def disagreedAccuracy(agreedVal):
    item_disagreed_val = []
    dis_acc = []
    for j in range(len(agreedVal)):
        coln = []
        for i in range(len(agreedVal[j])):
            if agreedVal[j][i] == 'disagreed':
                fin_1 = finVal[j][i], test_data[j][i]
                coln.append(fin_1)
        item_disagreed_val.append(coln)
    #Agreed Accuracy Caluclation
    total_counter =[]
    counter = []
    for j in range(len(item_disagreed_val)):
        count = 0
        total_count = 0
        for i in range(len(item_disagreed_val[j])):
                if item_disagreed_val[j][i][0] == item_disagreed_val[j][i][1]:
                    count= count +1
                    total_count = total_count + 1
                else:
                    total_count = total_count + 1
        counter.append(count)
        total_counter.append(total_count)

    #print final_count
    #print counter
    #print total_counter
    for i in range(len(counter)):
        dis_acc.append(float(counter[i])/total_counter[i])
    return dis_acc

#Function to calculate the accuracy of CL2 when the samples disagreed
def disagreed_CL2_accuracy(agreedVal):
    dis_accr = []
    user_disagreed_val = []
    for j in range(len(agreedVal)):
        coln = []
        for i in range(len(agreedVal[j])):
            if agreedVal[j][i] == 'disagreed':
                fin_1 = user_finVal[j][i], test_data[j][i]
                coln.append(fin_1)
        user_disagreed_val.append(coln)
    #Agreed Accuracy Caluclation
    total_counter =[]
    counter = []
    for j in range(len(user_disagreed_val)):
        count = 0
        total_count = 0
        for i in range(len(user_disagreed_val[j])):
                if user_disagreed_val[j][i][0] == user_disagreed_val[j][i][1]:
                    count= count +1
                    total_count = total_count + 1
                else:
                    total_count = total_count + 1
        counter.append(count)
        total_counter.append(total_count)

    #print final_count
    print counter
    print total_counter
    for i in range(len(counter)):
        dis_accr.append(float(counter[i])/total_counter[i])
    return dis_accr

agreementPercentage()
agreedAccuracy()
disagreedAccuracy()
disagreed_CL2_accuracy()