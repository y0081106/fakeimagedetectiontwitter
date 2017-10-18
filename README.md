# fakeimagedetectiontwitter
A system that learns how to detect fake images on twitter.

userFeaturesGeneration.py and itemFeaturesGeneration.py extract User Features and Tweet Features from the tweets.
userFeaturesClassification.py and itemFeaturesClassification.py perform the classification for the respecitive features.
agreementRetraining.py performs agreement based retraining classification and gives the final result.
fake_tweets.json and real_tweets.json contain the fake and real tweets collected from 3 terrorist attacks - Berlin 2016, Stockholm 2017 and Manchester 2017.

Download these files to calculate the following functions:
Harmonic Centrality: http://data.dws.informatik.uni-mannheim.de/hyperlinkgraph/2012-08/ranking/hostgraph-h.tsv.gz
Indegree: http://data.dws.informatik.uni-mannheim.de/hyperlinkgraph/2012-08/ranking/hostgraph-indegree.tsv.gz
