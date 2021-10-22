#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vyvo
"""

import LC
from cleantext import *

def MTLD(token_list, factor_size):
    
    length = len(token_list)
    factor_cut = 0
    factor_count = 0 
    types = set() 
    
    # factorize
    for i in range(length):
        types.add(token_list[i])
        TTR = len(types)/(i+1-factor_cut)
        if TTR <= factor_size:
            factor_cut = i+1 
            factor_count += 1
            types = set()
    
    # calculate remainder score 
    if TTR > factor_size: 
        IFS = (1-TTR)/(1-factor_size)
        factor_count += IFS
    
    return length/factor_count 


class VText(object):
    
    def __init__(self, text):
        self.content = text.lower()
    
    def clean_content(self, stopword=0, lemmatize=1, keepchar=None, to_tokens=False):
        clean_text = quick_clean_text(self.content, html=0, stopword=stopword, 
                                      lemmatize=lemmatize, keepchar=keepchar, to_tokens=to_tokens)
        
        return clean_text
    
    def extract_sentences(self, into_tokens=False, lemmatize=0):
        clean_text = self.clean_content(lemmatize=lemmatize, keepchar='.')
        sentences = re.findall(r"\.?(.{3,}?)\.", clean_text)
        if into_tokens:
            return [tokenizer.tokenize(s) for s in sentences]
        else: 
            return sentences
    
    def word_count(self):
        clean_tokens = self.clean_content(lemmatize=0, to_tokens=True)
        return len(clean_tokens)
    
    def sentence_count(self):
        clean_sentences = self.extract_sentences()
        return len(clean_sentences)
    
    def tag_pos(self, level='entire', lemmatize=0): 
        assert level == 'entire' or level == 'sentence'
        if level == 'entire':
            clean_tokens = self.clean_content(lemmatize=lemmatize, to_tokens=True)
            token_tag = nltk.pos_tag(clean_tokens)
        else: 
            clean_sentences = self.extract_sentences(into_tokens=True, lemmatize=lemmatize)
            token_tag = [nltk.pos_tag(s) for s in clean_sentences]
        return token_tag
    
    def extract_pos(self, pos, level='entire', lemmatize=0):
        assert level == 'entire' or level == 'sentence'
        token_tag = self.tag_pos(level, lemmatize)
        if level == 'entire':
            filtered_words = [tags[0] for tags in token_tag if tags[1] in pos]
        else: 
            filtered_words = []
            for s in token_tag:
                filtered_words.append([tags[0] for tags in s if tags[1] in pos])
        
        return filtered_words
    
    def extract_instance(self, word_pos):
        """
        Return sentence-annotated instances, each instance is a tuple (word, sentence)
        """
        filtered_words = self.extract_pos(word_pos, 'sentence')  
        
        n = len(filtered_words)
        instances = {}
    
        for i in range(n):
            for token in filtered_words[i]:
                if token not in instances: 
                    instances[token] = []
                instances[token].append((token,i))
                    
        return instances
    
    def LDscore(self, factor_size=0.72):
        token_list = self.clean_content(to_tokens=True)
        forward = MTLD(token_list, factor_size)
        reverse = MTLD(token_list[::-1], factor_size)
        return (forward+reverse)/2 
    
    def average_word_frequency(self):
        token_list = self.clean_content(stopword=1, lemmatize=0, to_tokens=True)
        type_list = set(token_list)
        
        # Loading Brown corpus
        from nltk.corpus import brown
        nltk.download('brown')
        fdist = nltk.FreqDist(brown.words())
        
        freq = 0 
        for t in type_list:
            freq += fdist[t]
        
        return freq/len(type_list)
            

    def LChain_features(self, LChain):
        f1 = len(LChain)
        f2 = LChain.average_chain_length()
        f3 = LChain.average_chain_span()
        f4 = LChain.count_cross_chains()
        
        # Number of half-document span chains
        f5 = (LChain.chain_span() > (self.sentence_count()/2)).sum()
        return np.array([f1,f2,f3,f4,f5])
        
        
        
        
        
        