#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (C) Apr 12, 2014 Han Zhao <han.zhao@uwaterloo.ca>
'''
@purpose: Base class for Hidden Markov Model
@author: Han Zhao (Keira)
'''
import cPickle
import numpy as np
from pprint import pprint
   
class SLHMM:
    
    _num_hidden = None
    _num_hidden = None
    def __init__(self, num_hidden, num_observ, 
                 transition_matrix=None, observation_matrix=None, initial_dist=None):
        self._num_hidden = num_hidden
        self._num_observ = num_observ
        
        self._P_1 = np.zeros(self._num_observ, dtype=np.float)
        self._P_21 = np.zeros((self._num_observ, self._num_observ), dtype=np.float)
        self._P_3x1 = np.zeros((self._num_observ, self._num_observ, self._num_observ), dtype=np.float)


    # Override the fit algorithm provided in HMM, using Spectral Learning 
    # algorithm
    def fit(self, sequences, rank_hyperparameter=None, verbose=False):
        
        # Set default value of rank-hyperparameter
        if rank_hyperparameter == None:
            rank_hyperparameter = self._num_hidden
        # Training triples
        trilst = np.array([sequence[idx: idx+3] for sequence in sequences
                           for idx in xrange(len(sequence)-2)], dtype=np.int)
        if verbose:
            pprint('Number of separated triples: %d' % trilst.shape[0])
        # Parameter estimation
        # Frequency based estimation
        for sq in trilst:
            self._P_1[sq[0]] += 1
            self._P_21[sq[1], sq[0]] += 1
            self._P_3x1[sq[1], sq[2], sq[0]] += 1
        # Normalization of P_1, P_21, P_3x1
        norms = np.sum(self._P_1)
        self._P_1 /= norms
        # Normalize the joint distribution of P_21        
        norms = np.sum(self._P_21)
        self._P_21 /= norms
        # Normalize the joint distribution of P_3x1
        norms = np.sum(self._P_3x1)
        self._P_3x1 /= norms
        # Singular Value Decomposition
        # Keep all the positive singular values
        (U, _, V) = np.linalg.svd(self._P_21)
        U = U[:, 0:rank_hyperparameter]
        V = V[0:rank_hyperparameter, :]
        # Compute b1, binf and Bx
        # self.factor = (P_21^{T} * U)^{+}, which is used to accelerate the computation
        factor = np.linalg.pinv(np.dot(self._P_21.T, U))
        self._b1 = np.dot(U.T, self._P_1)        
        self._binf = np.dot(factor, self._P_1)
        self._Bx = np.zeros((self._num_observ, rank_hyperparameter, rank_hyperparameter), dtype=np.float)        
        for index in xrange(self._num_observ):
            tmp = np.dot(U.T, self._P_3x1[index])
            self._Bx[index] = np.dot(tmp, factor.T)

    # Overwrite the prediction algorithm using DP provided in base class
    def predict(self, sequence):
        
        prob = self._b1
        for ob in sequence:
            prob = np.dot(self._Bx[ob], prob)
        prob = np.dot(self._binf.T, prob)
        return prob

    def get_rankings(self,seq):
        probs = []
        seq.append(0)
        for i in range(0,self._num_observ):
            seq[len(seq) - 1] = i
            
            probs.append(self.predict(seq))

        r = np.argsort(probs)[::-1]
        r[r == self._num_observ - 1]  = -1

        return r


