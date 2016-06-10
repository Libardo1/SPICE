import numpy as np
import itertools as it
class SpectralLearn :
    problemfile = ""
    order = 0
    p21 = None
    p31 = []
    sym_cnt = 0
    U = None
    V = None
    Sig = None
    Bx = []
    Sig_inv = None
    b0 = None
    binf = None
    p1 = None
    
    def __init__(self, probfile, ord):
        self.problemfile = probfile
        self.order = ord
    
    def initialize(self, line):
        line = line.strip(" ").split(" ")
        self.sym_cnt = int(line[1])
        self.sym_cnt +=1
        self.p1 = np.zeros((1,self.sym_cnt))
        self.p21 = np.zeros((self.sym_cnt,self.sym_cnt))
        
        for i in range(0,self.sym_cnt) :
            self.p31.append(np.zeros((self.sym_cnt,self.sym_cnt)))
        
    def learn(self):
        f = open(self.problemfile, "r")
        for i,line in enumerate(f):
            if(i==0):
                self.initialize(line)
                continue
            
            obs = [int(j) for j  in line.strip(" ").split(" ")]
            
            obs.append(self.sym_cnt -1)
            self.addToP1(obs[1:])
            self.addToP21(obs[1:])
            self.addToP31(obs[1:])
            
        self.normalize()
        self.U, self.Sig, self.V = self.svdP21()
        sig = np.zeros((len(self.Sig),len(self.Sig)))
        for i in range(0,len(self.Sig)):
            sig[i][i] = self.Sig[i]
        self.Sig = sig
        self.Sig_inv = np.linalg.inv(self.Sig)
        
        self.computeBx()
        self.computeB0_Binf()
        
    
    def addToP1(self, obs):
        for i in obs:
            self.p1[0][i] += 1
            
    def addToP21(self, obs):
        for first, second in it.izip(obs, obs[1:]):
            self.p21[first][second] += 1
        
    def addToP31(self,obs):
        for first, second, third in it.izip(obs, obs[1:], obs[2:]):
            self.p31[second][first][third] += 1
    
    def normalize(self):
        if(1):
            self.p21 += 0.0000000001
        self.p1 = self.p1 / np.sum(self.p1)
        self.p21 = self.p21 / np.sum(self.p21)
        for i,m in enumerate(self.p31):
            sum_m = np.sum(m)
            if(sum_m == 0.0):
                continue
            self.p31[i] = m / sum_m
            
    def svdP21(self):
        return np.linalg.svd(self.p21)
        
    def computeBx(self):
        for i,p31 in enumerate(self.p31):
            a = np.dot(np.dot(np.dot(self.U , p31) , self.V) , self.Sig_inv)
            self.Bx.append(a)
             
    
    def computeB0_Binf(self):
        self.b0 =  np.dot(self.U , self.p1[0].T)
        np.linalg.inv(np.dot(self.p21.T , self.U ))
        self.binf = np.dot(np.linalg.inv(np.dot(self.p21.T , self.U )) , self.p1.T)    
                 
    
    def predictRanks(self, t_obs):
        rank = []
        temp_lik = self.b0
        for o in t_obs:
            temp_lik = np.dot(temp_lik , self.Bx[o])
        for i in range(self.sym_cnt):
            t = np.dot(np.dot(temp_lik , self.Bx[0]) , self.binf)
            print(t)
            rank.append(t )
            
        print(rank)
        r = np.argsort(rank)[::-1]
        r[r == self.sym_cnt -1]  = -1
        
        return r,rank
        
            
         
            
        
    
    
        
                
            
                
        