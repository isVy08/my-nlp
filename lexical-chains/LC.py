#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vyvo
"""

# Imports
import numpy as np
from nltk.corpus import wordnet as wn


def get_word_senses(word, synset_pos=None):
    """
    Return a set of synsets, direct hypernyms of all synsets and hyponyms of a word 
    
    """
    syn = wn.synsets(word, synset_pos) # return all part-of-speeches if pos is None
    hyper = set()
    hypo = set()
    for s in syn:
        hyper.update(s.hypernyms())
        hypo.update(s.hyponyms())
    return set(syn), hyper, hypo

def get_text_senses(instances, synset_pos=None):
        """ 
        Return senses and candidate instances of an essay based on its instances 
        """
        words = set(instances.keys())
        senses = {}
        
        for w in words:
            syn = wn.synsets(w, synset_pos)
            for s in syn: 
                if s not in senses:
                    senses[s] = []
                
                cands = set(s.lemma_names()) & words
                if len(cands) > 0:
                    for c in cands:
                        senses[s].extend(instances[c])
        return senses
    
    
    
def get_relation(synset, candidates, instance, synset_pos=None):
    """
    Return the type of relation between a synset with an instance:
        
    - relation = 1: Identity or Synonym
        One of the candidate instances
    
    - relation = 2: Hypernym / Hyponym
        Hypernym / Hyponym 1: The synset is one of hypernyms / hyponyms of the instance word
        Hypernym / Hyponym 2: One of the synsets of the instance word is a hypernym of the synset
        
    - relation = 3: Siblings 
        The (synsets of) instance word and the synset share a common hypernym 
        
    - relation = 0: No relation 
        Otherwise
    """
    
    if instance in candidates: # identity or synonyms
        return 1
    else: 
        word_syn, word_hyper, word_hypo = get_word_senses(instance[0], synset_pos) 
        if synset in word_hyper or synset in word_hypo: # hypernym 1 / hyponym 1
            return 2 
        else: 
            syn_hyper = set(synset.hypernyms())
            if len(syn_hyper & word_syn) > 0: # hypernym 2 
                return 2
            elif len(syn_hyper & word_hyper) > 0: # siblings 
                return 3
            else: 
                syn_hypo = set(synset.hyponyms())
                if len(syn_hypo & word_syn) > 0: # hyponym 2 
                    return 2
                else:
                    return 0 

def i2i_weight(target, candidate, relation=0):
    """
    Return the weight of relation between an instance and a candidate instance of a synset, 
    depending on the distance (measured by sentences) 
    and type of relation between the instance and the synset containing that candidate instance
    """
    if target[0] == candidate[0]: # weight = 1 if two instances are duplicates
        return 1
    
    d = abs(target[1]-candidate[1])
    if relation == 1: 
        if d >= 5:
            return 0.5
        else: 
            return 1
    
    elif relation == 2: 
        if d >= 5: 
            return 0.3
        elif d >= 3:
            return 0.5
        else: 
            return 1
    else: 
        if d >= 7:
            return 0
        elif d >= 5:
            return 0.2
        elif d >= 3:
            return 0.3
        else: 
            return 1

            
def i2s_link(synset, candidates, instance):
    """
    Return the type and weight of relation between an instance and a synset.
    It is the sum of weights of relation between the instance and each candidate instance.
    """
    
    total_weight = 0
    relation = get_relation(synset, candidates, instance)
    if relation > 0:
        for other in candidates:
            total_weight += i2i_weight(instance, other, relation)
    return relation, total_weight


def restrict_unique_sense(single_word_weight):
    """
    Return the list of weights 
    """
    
    max_weight = np.max(single_word_weight)
    max_index = np.argmax(single_word_weight)
    n = len(single_word_weight)
    
    l = np.concatenate((np.repeat(0,max_index),
                        max_weight,
                        np.repeat(0,n-max_index-1)),
                       axis=None) 
    return l


        
class LC(object):
    
    def __init__(self, VText, word_pos, threshold, synset_pos=None):
        
        """Build instance-based chains"""
        
         # Initialize attributes
        self.instances = VText.extract_instance(word_pos)
        senses = get_text_senses(self.instances, synset_pos)
        
        # Group related instances                         
        weights = []
        for synset, candidates in senses.items(): 
            synset_weight = []
            for word, occurences in self.instances.items():
                word_weight = 0
                for inst in occurences:
                    relation, inst_weight = i2s_link(synset, candidates, inst)
                    if relation > 1:
                        senses[synset].append(inst)
                    word_weight += inst_weight
            
                synset_weight.append(word_weight)
            weights.append(synset_weight)
        
        
        # Prune unrelated chains 
        word_weight = np.array([restrict_unique_sense(x) for x in (np.array(weights).T)])
        final_chain = {}
        instance_value = list(self.instances.values())
        sense_name = list(senses.keys())
        
        
        for i, x in enumerate(word_weight.T):
            if np.sum(x) >= threshold:
                final_chain[sense_name[i]] = []
                selected = np.argwhere(x>0).reshape(1,-1)[0] 
                for j in selected:
                    final_chain[sense_name[i]].extend(instance_value[j])
        

        # Get final instance-based chain
        self.instance_chain = final_chain

        
        # Get word and sentence chains
        word_chain = {}
        sentence_chain ={}
        for syn, candidates in final_chain.items():
            word_chain[syn] = set()
            sentence_chain[syn] = set()
            for inst in candidates: 
                word_chain[syn].add(inst[0])
                sentence_chain[syn].add(inst[1])
        
        self.word_chain = word_chain
        self.sentence_chain = sentence_chain
        
        self.nodes = list(final_chain.keys())

        
     
    def __len__(self):
        return len(self.nodes)
    
    def chain_length(self):
        chain_length = np.array([len(c) for c in self.instance_chain.values()])
        return chain_length
    
    def average_chain_length(self):
        return self.chain_length().mean()
    
    def chain_span(self):
        chain_span = []
        for chain in self.instance_chain.values():
            l = [inst[1] for inst in chain]
            chain_span.append(l[-1] - l[0])
        return np.array(chain_span)
        
    
    def average_chain_span(self): 
        # measured in sentences
        return self.chain_span().mean()
    
    
    def count_cross_chains(self):
        # chains with overlapping sentences
        count = 0
        s = list(self.sentence_chain.values())
        for i in range(len(self)-1):
            if len(s[i] & s[i+1]) > 0: 
                count += 1
        return count

    
    def describe_chain(self):
        for syn, word in self.word_chain.items():
            print(syn,':',syn.definition())
            print('---', word)
            print('\n')
        
                 
        
        