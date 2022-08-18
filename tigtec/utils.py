import matplotlib.pyplot as plt
import math 
import os
import pandas as pd
import numpy as np

from copy import deepcopy
import string  

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

#Granular text processing functions
def lower(x:str):
    return str(x).lower()


def apostroph(x:str):
    """Remove apostrophs and replace them by a more explicite part of sentence
    Args:
        x (str): input string to transform
    Returns
        str - the transformed string
    """
    x = x.replace("n’t", " not")
    x = x.replace("’s", " is")
    x = x.replace("’m", " am")
    x = x.replace("’ve", " have")
    
    x = x.replace("n't", " not")
    x = x.replace("'s", " is")
    x = x.replace("'m", " am")
    x = x.replace("'ve", " have")

def remove_punct(x:str):
    """Remove punctation from a string

    Args:
        x (str): input string to transform

    Returns
        str - the transformed string
    """
    return x.translate(str.maketrans('', '', string.punctuation))

def remove_multi_space(x:str):
    """Remove multiple spaces from a string

    Args:
        x (str): input string to transform

    Returns
        str - the transformed string
    """
    return " ".join(x.split())


def text_to_token(x, nlp, lemmatize = True, stop_words = False):
    if lemmatize:
        if stop_words == False:
            return [token.lemma_ for token in nlp(x)]
        elif stop_words == 'spacy':
            return [token.lemma_ for token in nlp(x) if not(token.is_stop)]
        else:
            return [token.lemma_ for token in nlp(x) if not(token.lemma_ in stop_words)]

def lowering(clean_text):
        """
        lower the case of self.clean_text
        """
        clean_text = map(clean_text, lower)
        return clean_text
    
def apostrophs(clean_text):
        """
        replace apostrophs part in english clean_text by their developped form ex : don't --> do not; I've --> I have
        """
        clean_text = map(clean_text, apostroph)
        return clean_text
    
def remove_puncts(clean_text):
        """
        remove all form of punctuation in the clean_text
        """
        clean_text = map(clean_text, remove_punct)
        return clean_text

def remove_multi_spaces(clean_text):
        """
        remove double space in clean_text
        """
        clean_text = map(clean_text, remove_multi_space)
        return clean_text

def remove_unis(clean_text):
        """
        remove unicode characters in clean_text
        """
        clean_text = map(clean_text, remove_uni)
        return clean_text    

    
def text_to_tokens(x,nlp, lemmatize = True):
        """
        transform a sentence into an array of tokens
        """
        return text_to_token(x, nlp, lemmatize, stop_words)    
    
def remove_brbr_imdb(x:str) :
    """remove "br" string noise

    Args:
        x (string): text to correct
    """
    x = x.replace("br br", "")
    return(x)
    
def remove_brbrs_imdb(text) :
    """remove "br" string noise on all the texts

    Args:
        text (string pandas serie): text serie to correct
    """
    text = map(text, remove_brbr_imdb)
    return(text)

def clean_text(text, data='imdb'):
    """launch all the text cleaning process

    Args:
        text (string pandas serie): text to clean
        data (str, initial data specification): data specific correction. Defaults to 'imdb'.

    Returns:
        _type_: cleaned text
    """
    text = deepcopy(text)
    text = lowering(text)
    text = remove_multi_spaces(text)
    text = remove_puncts(text)
    if (data == 'imdb') :
        text = remove_brbrs_imdb(text) 
    return text