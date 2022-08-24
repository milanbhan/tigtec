import matplotlib.pyplot as plt
import math 
import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

from copy import deepcopy
import string  

import re
from langdetect import detect, DetectorFactory


import snscrape
import snscrape.modules.twitter as sntwitter

#Text Processing
def map(X, func):
        '''
        Transform the whole corpus based on a function made for one line
        input : 
            X : one element of the corpus
            func : the function you want to apply
        '''
        
        X1 = list(np.zeros(len(X)))
        for i, elt in enumerate(tqdm(X)):
            X1[i] = func(elt)
        return X1

def clean_tweet(tweet) :
    tweet = re.sub(r"\n", " ", tweet)
    tweet = re.sub(r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b\S+","",tweet)
    tweet = re.sub(r"bit.ly\S+", "", tweet)
    tweet = re.sub(r"twitter.com\S+", "", tweet)
    tweet = re.sub(r"@", "", tweet)
    tweet = re.sub(r'"', "", tweet)
    tweet = re.sub(r"#", "", tweet)
    return(tweet)

def remove_url(tweet) :
    new_text = []
    for t in tweet.split(" "):
        t = '' if t.startswith('https:') else t
        new_text.append(t)
    return " ".join(new_text)

def remove_hashtag(tweet) :
    new_text = []
    for t in tweet.split(" "):
        t = '' if t.startswith('#') else t
        new_text.append(t)
    return " ".join(new_text)

def get_language(text) :
    DetectorFactory.seed = 0
    try :
        return(detect(text))
    except :
        pass
  
def remove_emoji(text) :
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "]+", flags=re.UNICODE)
    return(emoji_pattern.sub(r'', text)) # no emoji


def remove_mlp_clue(text) :
    text = re.sub(r"MLP", " ", text)
    text = re.sub(r"MarinePrésidente ", " ", text)
    text = re.sub(r"Marine", " ", text)
    text = re.sub(r"Le Pen", " ", text)
    
    return(text)

def remove_zemmour_clue(text) :
    text = re.sub(r"Eric Zemmour", " ", text)
    text = re.sub(r"MerciZemmour", " ", text)
    text = re.sub(r"Zemmour", " ", text)
    text = re.sub(r"eric zemmour", " ", text)
    text = re.sub(r"RECONQUETE", " ", text)
    return(text)

def remove_macron_clue(text) :
    text = re.sub(r"Emmanuel Macron", " ", text)
    text = re.sub(r"EMacron", " ", text)
    text = re.sub(r"Macron", " ", text)
    text = re.sub(r"emmanuel macron", " ", text)
    return(text)

def remove_melanchon_clue(text) :
    text = re.sub(r"Melenchon", " ", text)
    text = re.sub(r"JLMelenchon", " ", text)
    text = re.sub(r"UnionPopulaire", " ", text)
    text = re.sub(r"JLM", " ", text)
    return(text)

def remove_jadot_clue(text) :
    text = re.sub(r"yjadot", " ", text)
    text = re.sub(r"Jadot2022", " ", text)
    text = re.sub(r"JeunesJadot", " ", text)
    text = re.sub(r"Jeunesetjadot", " ", text)
    text = re.sub(r"Primaireecolo", " ", text)
    text = re.sub(r"Primaireécologiste", " ", text)
    text = re.sub(r"Jadot", " ", text)
    return(text)

def remove_stop_word(text, stop_word) :
    for t in stop_word :
        text = re.sub(t, " ", text)
  
    return(text)

def remove_bad_space(text) :
    #Enlever espace au début
    text = text.lstrip()
    #Enlever espace à la fin
    text = text.rstrip()
    #Enlever espace duppliqué
    text = text.split()
    text = " ".join(text)

    return(text)

def filter_and_clean_twitt(df, text:str, target:str, stop_word:list) :
    #Sélection du texte français seulement
    df["language"]=df[text].apply(lambda t : get_language(t))
    df = df[df.language =='fr']
    
    #Filtrer les twitts 
    df = df[df[text].isin(["''"," "])==False]
    df = df[df[text].isnull()==False]
    df = df[df[text].str.len()>35]
    
    #Nettoyage text
    df[text] = df[text].apply(lambda x : remove_url(x))
    df[text] = df[text].apply(lambda x : clean_tweet(x))
    df[text] = df[text].apply(lambda x : remove_emoji(x))
    
    #Suppression des signatures "MLP", c'est de la triche
    df[text][df[target]=='MLP_officiel'] = df[text][df[target]=='MLP_officiel'].apply(lambda x : remove_mlp_clue(x))
    df[text][df[target]=='ZemmourEric'] = df[text][df[target]=='ZemmourEric'].apply(lambda x : remove_zemmour_clue(x))
    df[text][df[target]=='EmmanuelMacron'] = df[text][df[target]=='EmmanuelMacron'].apply(lambda x : remove_macron_clue(x))
    df[text][df[target]=='JLMelenchon'] = df[text][df[target]=='JLMelenchon'].apply(lambda x : remove_melanchon_clue(x))
    df[text][df[target]=='yjadot'] = df[text][df[target]=='yjadot'].apply(lambda x : remove_jadot_clue(x))
    
    #Remove stop_words and bad spaces
    df[text]= df[text].apply(lambda x : remove_stop_word(x,stop_word))
    df[text]= df[text].apply(lambda x : remove_bad_space(x))
    
    return(df)

def scrap_twitter (candidate_list, nb_tweets) :
    tweets_list1 = []
    for c in candidate_list :
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'from:{c}').get_items()): #declare a username 
            if i>nb_tweets: #number of tweets you want to scrape
                break
        tweets_list1.append([tweet.date, tweet.id, tweet.content, tweet.user.username]) #declare the attributes to be returned

    # Creating a dataframe from the tweets list above 
    tweets_df = pd.DataFrame(tweets_list1, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])

    #Enlever les na
    tweets_df = tweets_df[tweets_df.Username.isin(candidate_list)]
    
    return(tweets_df)

            

