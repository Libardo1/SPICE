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

class HMM(object):
    '''
    Hidden Markov Model (HMM) for multinomial observation sequence.
    
    Provide basic interface to be implemented by different training algorithms, 
    including but not limited to Expectation Maximization, Spectral Learning, etc.
    In general, HMM can be used to solve three kinds of problems:
    1,     Estimation problem:
           Given an observable sequence, o1, o2, ..., ot, computing the marginal 
           probability Pr(o1, o2, ..., ot).
    
    2,     Decoding problem:
           Given an observable sequence, o1, o2, ..., ot, infer the hidden state sequence
           s1, s2, ..., st which gives the largest probability for o1, o2, ..., ot, 
           i.e., decoding problem solves the following problem:
           s1, s2, ..., st = argmax Pr(o1, o2, ..., ot|s1, s2, ..., st)
    
    3,     Learning problem:
           Given a set of observation sequences, infer for the transitioin matrix, observation
           matrix and initial distribution, i.e., learning problem solves the following problem:
           T, O, pi = argmax Pr(X | T, O, pi)
           where X is the set of observation sequences.
    '''
    def __init__(self, num_hidden, num_observ,
                 transition_matrix=None, observation_matrix=None, initial_dist=None):
        '''
        @num_hidden: np.int. Number of hidden states in the HMM.
        @num_observ: np.int. Number of observations in the HMM. 
        
        @transition_matrix: np.ndarray, shape = (num_hidden, num_hidden)
                            Transition matrix of the HMM, denoted by T, 
                            T_ij = Pr(h_t+1 = i | h_t = j)
                            
                            Default value is None. If it is not None, it must satisfy
                            the following two conditions:
                            1.     shape = (num_hidden, num_hidden)
                            2.     All the elements should be non-negative
                            
                            Note: The input transition matrix will be normalized column-wisely
        @observation_matrix: np.ndarray, shape = (num_observ, num_hidden)
                             Observation matrix of the HMM, denoted by O,
                             O_ij = Pr(o_t = i | h_t = j)
                            
                            Default value is None. If it is not None, it must satisfy 
                            the following two conditions:
                            1.        shape = (num_observ, num_hidden)
                            2.        All the elements should be non-negative
                            
                            Note: The input observation matrix will be normalized column-wisely
        @initial_dist: np.ndarray, shape = (num_hidden,)
                       Initial distribution for hidden states.
                       Pi_i = Pr(h_1 = i)
                       
                       Default value is None. If it is not None, it must satisfy the following two
                       conditions:
                       1.     shape = (num_hidden,)
                       2.     All the elements should be non-negative
                       
                       Note: The input array will be normalized to form a probability distribution.
        '''
        if num_hidden <= 0 or not isinstance(num_hidden, int):
            raise ValueError("Number of hidden states should be positive integer")
        if num_observ <= 0 or not isinstance(num_observ, int):
            raise ValueError("Number of observations should be positive integer")
        self._num_hidden = num_hidden
        self._num_observ = num_observ
        # Build transition matrix, default is Identity matrix 
        if transition_matrix != None:
            if not (transition_matrix.shape == (num_hidden, num_hidden)):
                raise ValueError("Transition matrix should have size: (%d, %d)" 
                                 % (num_hidden, num_hidden))
            if not np.all(transition_matrix >= 0):
                raise ValueError("Elements in transition matrix should be non-negative")
            self._transition_matrix = transition_matrix
            norms = np.sum(transition_matrix, axis=0)
            self._transition_matrix /= norms
        else:
            self._transition_matrix = np.eye(num_hidden, dtype=np.float)
        # Build observation matrix, default is Identity matrix
        if observation_matrix != None:
            if not (observation_matrix.shape == (num_observ, num_hidden)):
                raise ValueError("Observation matrix should have size: (%d, %d"
                                 % (num_observ, num_hidden))
            if not np.all(observation_matrix >= 0):
                raise ValueError("Elements in observation matrix should be non-negative")
            self._observation_matrix = observation_matrix
            norms = np.sum(observation_matrix, axis=0)
            self._observation_matrix /= norms
        else:
            self._observation_matrix = np.eye(num_observ, num_hidden)
        # Build initial distribution, default is uniform distribution
        if initial_dist != None:
            if not (initial_dist.shape[0] == num_hidden):
                raise ValueError("Initial distribution should have length: %d" % num_hidden)
            if not np.all(initial_dist >= 0):
                raise ValueError("Elements in initial_distribution should be non-negative")
            self._initial_dist = initial_dist
            self._initial_dist /= np.sum(initial_dist)
        else:
            self._initial_dist = np.ones(num_hidden, dtype=np.float)
            self._initial_dist /= num_hidden
        # Build accumulative Transition matrix and Observation matrix, which will
        # be useful when generating observation sequences
        self._accumulative_transition_matrix, self._accumulative_observation_matrix = \
        np.add.accumulate(self._transition_matrix, axis=0), \
        np.add.accumulate(self._observation_matrix, axis=0)
        
    ##############################################################################
    # Public methods
    ##############################################################################        
    @property
    def initial_dist(self):
        return self._initial_dist
    
    @property
    def transition_matrix(self):
        return self._transition_matrix
    
    @property
    def observation_matrix(self):
        return self._observation_matrix

    def decode(self, sequence):
        '''
        Solve the decoding problem with HMM, also called "Viterbi Algorithm"
        @sequence: np.array. Observation sequence
        @attention: Note that all the observations should be encoded as integers
                    between [0, num_observ)
        @note: Using dynamic programming to find the most probable hidden state
               sequence, the computational complexity for this algorithm is O(Tm^2),
               where T is the length of the sequence and m is the number of hidden
               states.
        '''
        t = len(sequence)
        prob_grids = np.zeros((t, self._num_hidden), dtype=np.float)
        path_grids = np.zeros((t, self._num_hidden), dtype=np.int)
        # Boundary case
        prob_grids[0, :] = self._initial_dist * self._observation_matrix[sequence[0], :]
        path_grids[0, :] = -1
        # DP procedure, prob_grids[i, j] = max{prob_grids[i-1, k] * T_{j,k} * O_{seq[i],j}}
        # Forward-computing of DP procedure
        for i in xrange(1, t):
            # Using vectorized code to avoid the explicit for loop, improve the efficiency,
            # i.e., H_k(i) = max{ H_{k-1}(j) * T_{i,j} * O_{seq[k], i}}, which can be formed
            # as a matrix by using outer product
            exp_prob = np.outer(self._observation_matrix[sequence[i], :], prob_grids[i-1, :])
            exp_prob *= self._transition_matrix
            prob_grids[i, :], path_grids[i, :] = \
            np.max(exp_prob, axis=1), np.argmax(exp_prob, axis=1)
        # Backward-path finding of DP procedure
        opt_hidden_seq = np.zeros(t, dtype=np.int)
        opt_hidden_seq[-1] = np.argmax(prob_grids[-1, :])
        for i in xrange(t-1, 0, -1):
            opt_hidden_seq[i-1] = path_grids[i, opt_hidden_seq[i]]
        return opt_hidden_seq
    
    def predict(self, sequence):
        '''
        Solve the estimation problem with HMM.
        @sequence: np.array. Observation sequence
        @attention: Note that all the observations should be encoded as integers
                    between [0, num_observ)
        '''
        return np.sum(self._alpha_process(sequence)[-1, :])

    def fit(self, sequences):
        '''
        Solve the learning problem with HMM.
        @sequences: [np.array]. List of observation sequences, each observation 
                    sequence can have different length
        @attention: Note that all the observations should be encoded as integers
                    between [0, num_observ)
        @note: This method should be overwritten by different subclasses. 
        '''
        pass

    def generate_data(self, dsize, min_seq_len=3, max_seq_len=50):
        '''
        Generate data based on the given HMM.
        @dsize: np.int. Number of observation sequences to be generated
        @min_seq_len: np.int. Minimum length of each observation sequence, inclusive
        @max_seq_len: np.int. Maximum length of each observation sequence, exclusive
        '''
        data = []
        for i in xrange(dsize):
            # Cumulative distribution of the states
            accdist = np.add.accumulate(self._initial_dist)
            rlen = np.random.randint(min_seq_len, max_seq_len)
            sq = np.zeros(rlen, dtype=np.int)
            # Initial state chosen based on the initial distribution
            state = np.where(accdist >= np.random.rand())[0][0]
            for j in xrange(rlen):
                # update the state of HMM by the Transition matrix[state]
                state = np.where(self._accumulative_transition_matrix[:, state] 
                                  >= np.random.rand())[0][0]
                # randomly choose an observation by the Observation matrix[state]
                observ = np.where(self._accumulative_observation_matrix[:, state] 
                                  >= np.random.rand())[0][0]
                sq[j] = observ
            data.append(sq)        
        return data
     
    #########################################################################
    # Protected methods
    #########################################################################
    def _alpha_process(self, sequence):
        '''
        @sequence: np.array. Observation sequence
        @note: Computing the forward-probability: Pr(o_1,o_2,...,o_t, h_t=i) 
               using dynamic programming. The computational complexity is O(Tm^2),
               where T is the length of the observation sequence and m is the 
               number of hidden states in the HMM.
        '''
        t = len(sequence)
        grids = np.zeros((t, self._num_hidden), dtype=np.float)
        grids[0, :] = self._initial_dist * self._observation_matrix[sequence[0], :]
        for i in xrange(1, t):
            grids[i, :] = np.dot(self._transition_matrix, grids[i-1, :])
            grids[i, :] *= self._observation_matrix[sequence[i], :]
        return grids
    
    def _beta_process(self, sequence):
        '''
        @sequence: np.array. Observation sequence
        @note: Computing the backward-probability: Pr(o_t+1, ..., o_T | h_t)
               using dynamic programming. The computational complexity is 
               O(Tm^2) where T is the length of the observation sequence and m
               is the number of hidden states in the HMM.
        '''
        t = len(sequence)
        grids = np.zeros((t, self._num_hidden), dtype=np.float)
        grids[t-1, :] = 1.0
        for i in xrange(t-1, 0, -1):
            grids[i-1, :] = grids[i, :] * self._observation_matrix[sequence[i], :]
            grids[i-1, :] = np.dot(grids[i-1, :], self._transition_matrix)
        return grids
    
    ######################################################
    # Static method
    ######################################################
    @staticmethod
    def to_file(filename, hmm):
        with file(filename, "wb") as fout:
            cPickle.dump(hmm, fout)
    
    @staticmethod
    def from_file(filename):
        with file(filename, "rb") as fin:
            model = cPickle.load(fin)
            return model
    


class SLHMM(HMM):
    '''
    This class is used to learning HMM using Spectral Learning algorithm.
    For more detail, please refer to the following paper:
    
    A Spectral Algorithm for Learning Hidden Markov Model, Hsu et al.
    http://arxiv.org/pdf/0811.4413.pdf
    
    Note that the spectral learning algorithm proposed in the paper above
    only supports solving the estimation problem and the learning problem 
    (not learning the transition matrix, observation matrix and initial
    distribution directly, but return a set of transformed observable 
    operators which support computing the marginal joint probability distribution:
    Pr(o_1, o_2,..., o_t))
    '''
    def __init__(self, num_hidden, num_observ, 
                 transition_matrix=None, observation_matrix=None, initial_dist=None):
        '''
        @num_hideen: np.int, number of hidden states in HMM.
        @num_observ: np.int, number of observations in HMM.
        
        @transition_matrix: np.ndarray, shape = (num_hidden, num_hidden)
                            Transition matrix of the HMM, denoted by T, 
                            T_ij = Pr(h_t+1 = i | h_t = j)
                            
                            Default value is None. If it is not None, it must satisfy
                            the following two conditions:
                            1.     shape = (num_hidden, num_hidden)
                            2.     All the elements should be non-negative
                            
                            Note: The input transition matrix will be normalized column-wisely
        @observation_matrix: np.ndarray, shape = (num_observ, num_hidden)
                             Observation matrix of the HMM, denoted by O,
                             O_ij = Pr(o_t = i | h_t = j)
                            
                            Default value is None. If it is not None, it must satisfy 
                            the following two conditions:
                            1.        shape = (num_observ, num_hidden)
                            2.        All the elements should be non-negative
                            
                            Note: The input observation matrix will be normalized column-wisely
        @initial_dist: np.ndarray, shape = (num_hidden,)
                       Initial distribution for hidden states.
                       Pi_i = Pr(h_1 = i)
                       
                       Default value is None. If it is not None, it must satisfy the following two
                       conditions:
                       1.     shape = (num_hidden,)
                       2.     All the elements should be non-negative
                       
                       Note: The input array will be normalized to form a probability distribution.
        '''
        # Call initial method of base class directly
        super(SLHMM, self).__init__(num_hidden, num_observ, 
                       transition_matrix, observation_matrix, initial_dist)
        # First three order moments
        self._P_1 = np.zeros(self._num_observ, dtype=np.float)
        self._P_21 = np.zeros((self._num_observ, self._num_observ), dtype=np.float)
        self._P_3x1 = np.zeros((self._num_observ, self._num_observ, self._num_observ), dtype=np.float)


    # Override the fit algorithm provided in HMM, using Spectral Learning 
    # algorithm
    def fit(self, sequences, rank_hyperparameter=None, verbose=False):
        '''
        Solve the learning problem with HMM.
        @sequences: [np.array]. List of observation sequences, each observation 
                    sequence can have different length
        
        @attention: Note that all the observations should be encoded as integers
                    between [0, num_observ). 
        '''
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
        '''
        Solve the estimation problem with HMM.
        @sequence: np.array. Observation sequence
        @attention: Note that all the observations should be encoded as integers
                    between [0, num_observ). This algorithm uses transformed 
                    observable operator to compute the marginal joint probability
        '''
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


