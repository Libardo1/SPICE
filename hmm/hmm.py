import numpy as np
import copy

class Hmm(object):

	def __init__(self, no_of_hidden_states, no_of_observed_states):
		self.no_of_hidden_states = no_of_hidden_states
		self.no_of_observed_states = no_of_observed_states+1
		
		# intialize emission density
		self.B = np.random.rand(self.no_of_hidden_states, self.no_of_observed_states)
		row_sums = self.B.sum(axis=1)
		self.B /=row_sums[:, np.newaxis]

		# intialize transition prob
		self.A = np.random.rand(self.no_of_hidden_states, self.no_of_hidden_states)
		row_sums = self.A.sum(axis=1)
		self.A /=row_sums[:, np.newaxis]

		self.pi = np.random.rand(self.no_of_hidden_states)
		self.pi /=self.pi.sum()


 
	def train(self,sequences,iterations=10):
 		updated_sequnece = [ np.asarray([s+1 for s in sequence]+[0]) for sequence in sequences]

 		for itr in xrange(iterations):
 			# print '===================='
 			print itr
 			A = self.A
	 		pi = self.pi
	 		B = self.B
	 		# print A,B,pi

	 		self.A = np.zeros((self.no_of_hidden_states,self.no_of_hidden_states))
	 		self.B = np.zeros((self.no_of_hidden_states,self.no_of_observed_states))
	 		self.pi = np.zeros(self.no_of_hidden_states)


 			for sequence in updated_sequnece:
 				sequence = np.asarray(sequence)
 				l = len(sequence)
				# alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
				# Initialize alpha
				alpha = np.zeros((self.no_of_hidden_states,l))
				c = np.zeros(l) #scale factors
				alpha[:,0] = pi.T * B[:,sequence[0]]
				c[0] = 1.0/np.sum(alpha[:,0])
				alpha[:,0] = c[0] * alpha[:,0]
				# Update alpha for each observation step
				for t in range(1,l):
					alpha[:,t] = np.dot(alpha[:,t-1].T, A).T * B[:,sequence[t]]
					c[t] = 1.0/np.sum(alpha[:,t])
					alpha[:,t] = c[t] * alpha[:,t]

				# beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
				# Initialize beta
				beta = np.zeros((self.no_of_hidden_states,l))
				beta[:,l-1] = 1
				beta[:,l-1] = c[l-1] * beta[:,l-1]
				# Update beta backwards from end of sequence
				for t in range(len(sequence)-1,0,-1):
					beta[:,t-1] = np.dot(A, (B[:,sequence[t]] * beta[:,t]))
					beta[:,t-1] = c[t-1] * beta[:,t-1]

				xi = np.zeros((self.no_of_hidden_states,self.no_of_hidden_states,l-1));
				for t in range(l-1):
					denom = np.dot(np.dot(alpha[:,t].T, A) * B[:,sequence[t+1]].T,beta[:,t+1])
					for i in range(self.no_of_hidden_states):
						numer = alpha[i,t] * A[i,:] * B[:,sequence[t+1]].T * beta[:,t+1].T
						xi[i,:,t] = numer / denom

				# gamma_t(i) = P(q_t = S_i | O, hmm)
				gamma = np.sum(xi,axis=1,keepdims=False)
				# Need final gamma element for new B


				prod =  (alpha[:,l-1] * beta[:,l-1]).reshape((-1,1))
				gamma = np.hstack((gamma,  prod / np.sum(prod))) #append one more to gamma!!!
				newpi = gamma[:,0]
				newA = np.sum(xi,2) / np.sum(gamma[:,:-1],axis=1).reshape((-1,1))
				newB = copy.copy(B)

				numLevels = B.shape[1]
				sumgamma = np.sum(gamma,axis=1)
				for lev in range(numLevels):
					mask = sequence == lev
					newB[:,lev] = np.sum(gamma[:,mask],axis=1) / sumgamma


				self.A +=newA
				self.B+=newB
				self.pi+=newpi

			row_sums = self.B.sum(axis=1)
			self.B /=row_sums[:, np.newaxis]

			# intialize transition prob
			row_sums = self.A.sum(axis=1)
			self.A /=row_sums[:, np.newaxis]

			self.pi /=self.pi.sum()

	# def train(self, sequences,iterations = 5 ):
	# 	updated_sequnece = [ [s+1 for s in sequence]+[0] for sequence in sequences]
		
	# 	for i in xrange(iterations):
	# 		print('iterations',i)
	# 		emission_count = np.zeros((self.no_of_hidden_states,self.no_of_observed_states))
	# 		transiztion_count =np.zeros((self.no_of_hidden_states,self.no_of_hidden_states))
	# 		initial_state_count = np.zeros(self.no_of_hidden_states)
	# 		for sequence in sequences:
	# 			_emission_count,_transiztion_count,_initial_state_count = self.E_step(sequence)
	# 			emission_count +=_emission_count
	# 			transiztion_count+=_transiztion_count
	# 			initial_state_count+=_initial_state_count
			
	# 		self.B = emission_count
	# 		row_sums = self.B.sum(axis=1)
	# 		self.B /=row_sums[:, np.newaxis]

	# 		# intialize transition prob
	# 		self.A = transiztion_count
	# 		row_sums = self.A.sum(axis=1)
	# 		self.A /=row_sums[:, np.newaxis]

	# 		self.pi = initial_state_count
	# 		self.pi /=self.pi.sum()

	# def E_step(self,sequence):
		
	# 	Ob = self.B
	# 	Tr = self.A
	# 	p0 = self.pi
	# 	o = sequence

	# 	dx,do = Ob.shape   # if a numpy matrix
	# 	L = len(o)
	# 	alpha = np.zeros((L,dx))
	# 	beta = np.zeros((L,dx))
	# 	weights = np.zeros((L,dx))
	# 	alpha[0,:] = p0*Ob[:,o[0]]    # compute initial forward message

	# 	for t in range(1,L):    # compute forward messages
	# 		alpha[t,:] = np.matmul(alpha[t-1,:],Tr)*Ob[:,o[t]]

	# 	beta[L-1,:] = np.ones(dx)  # initialize reverse messages
		
	# 	weights[L-1,:] = beta[L-1,:]*alpha[L-1,:]  # and marginals
	# 	weights[L-1,:] /= weights[L-1,:].sum()
		
	# 	for t in range(L-2,-1,-1):
	# 		beta[t,:] =  np.matmul(Tr,beta[t+1,:]*Ob[:,o[t+1]])
	# 		weights[t,:] = beta[t,:]*alpha[t,:]
	# 		weights[t,:] /= weights[t,:].sum()

	# 	emission_count = np.zeros((self.no_of_hidden_states,self.no_of_observed_states))
	# 	for t in range(0,L):
	# 		emission_count[:,o[t]] += weights[t,:]

	# 	gaama = np.zeros((L-1,dx,dx))

	# 	initial_state_count = weights[0,:]

	# 	for t in range(0,L-1):
	# 		# check if correct
	# 		a = beta[t,:]*Ob[:,sequence[t+1]]
	# 		a = a[:,np.newaxis].T
	# 		gaama[t,:,:] = a.T*Tr*alpha[t,:]
	# 		gaama[t,:,:] /=gaama[t,:,:].sum()
		
	# 	state_count = np.sum(weights,axis=0)
	# 	E = gaama.sum(axis=0)
	# 	transiztion_count = E/state_count[np.newaxis,:]

	# 	return emission_count,transiztion_count,initial_state_count
	
	def predict(self,sequence):
		alpha = self.pi

		if(len(sequence)==0):
			next_hidden_state_prob = self.pi
		else:
			alpha = self.pi*self.B[:,sequence[0]]
		
			if len(sequence)>1:
				for s in sequence[1:]:    # compute forward messages
					alpha = np.matmul(alpha,self.A)*self.B[:,s]

			next_hidden_state_prob = np.matmul(alpha,self.A)
		
		prob  = np.matmul(next_hidden_state_prob[np.newaxis,:] , self.B)
		return np.squeeze(np.argsort(prob)-1)[::-1]
		


