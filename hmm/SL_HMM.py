
import numpy as np   
class SLHMM:
    
    h_cnt = None
    sym_cnt = None
    p1 = None
    p21 = None
    p31 = None
    b1 = None
    binf = None
    bx = None
    U = None
    V = None
    Sig = None

    def initMatrices(self):
        self.p1 = np.zeros(self.sym_cnt)
        self.p21 = np.zeros((self.sym_cnt, self.sym_cnt))
        self.p31 = np.zeros((self.sym_cnt, self.sym_cnt, self.sym_cnt))


    def __init__(self, h_cnt, sym_cnt):
        self.h_cnt = h_cnt
        self.sym_cnt = sym_cnt  
        self.initMatrices() 
        
    def normalizePs(self):
        self.p1 = self.p1 / np.sum(self.p1)
        self.p21 = self.p21 / np.sum(self.p21)
        self.p31 = self.p31 / np.sum(self.p31)


    def computePs(self, triples):
        for s in triples:
            self.p1[s[0]] = self.p1[s[0]] + 1
            self.p21[s[1], s[0]] =self.p21[s[1], s[0]] + 1
            self.p31[s[1], s[2], s[0]] = self.p31[s[1], s[2], s[0]] + 1

    def computeBx(self, t):
        self.bx = np.zeros((self.sym_cnt, self.h_cnt, self.h_cnt))
        for i in range(0,self.sym_cnt):
            tmp = np.dot(self.U.T, self.p31[i])
            self.bx[i] = np.dot(tmp, t.T)


    def learn(self, sequences):
        triples = np.array([sequence[idx: idx+3] for sequence in sequences
                           for idx in xrange(len(sequence)-2)])
       
        self.computePs(triples)
        self.normalizePs()

        self.U, self.Sig, self.V = np.linalg.svd(self.p21)
        self.U = self.U[:, 0:self.h_cnt]
        self.V = self.V[0:self.h_cnt, :]

        temp = np.linalg.pinv(np.dot(self.p21.T, self.U))
        self.b1 = np.dot(self.U.T, self.p1)        
        self.binf = np.dot(temp, self.p1)

        self.computeBx(temp)

        

    # Overwrite the prediction algorithm using DP provided in base class
    def predict(self, sequence):
        
        prob = self.b1
        for ob in sequence:
            prob = np.dot(self.bx[ob], prob)
        prob = np.dot(self.binf.T, prob)
        return prob

    def get_rankings(self,seq):
        probs = []
        seq.append(0)
        for i in range(0,self.sym_cnt):
            seq[len(seq) - 1] = i
            
            probs.append(self.predict(seq))

        r = np.argsort(probs)[::-1]
        r[r == self.sym_cnt - 1]  = -1

        return r


