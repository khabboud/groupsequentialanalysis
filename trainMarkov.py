import seaborn as sns
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
class MarkovChain(object): 
    """ Train a discrete-time Markov chain without connection restrictions """
    
    def __init__(self):
        """ 
        constructor for class MarkovChain
        """
        self.states = [] 
        self.trans_mat = None
        self.freq_mat = None
        self.trans_dic = {} 
    
    def fit(self, data):    
        """ 
        Train a markov model with input sequence data
        
        Parameters:  data: sequence data in the form of list of lists
        
        """
        try:
            
            for seq in data: 

                prev_state = None 

                for state in seq: 

                    if prev_state: 

                        if (prev_state, state) in self.trans_dic.keys():
                            self.trans_dic[(prev_state, state)] +=1

                        else: self.trans_dic[(prev_state, state)] = 1

                    prev_state = state

            # using unique() of pandas O(N) complexity as compared to
            # numpy unique() which is O(NlogN) which returns sorted
            self.states = pd.unique([i[0] for i in self.trans_dic.keys()] + 
                                    [i[1] for i in self.trans_dic.keys()]) 

            self.num_states = len(self.states)

            self.freq_mat = np.zeros([self.num_states, self.num_states])

            # define dictionary to map states with their order 
            self.state_dic = {self.states[i]:i for i in range(0,self.num_states)}

            # construct frequency matrix
            for key in self.trans_dic.keys():
                self.freq_mat[self.state_dic[key[0]],self.state_dic[key[1]]] = self.trans_dic[key] 

            # derive transition matrix
            freq_sum = np.sum(self.freq_mat, axis = 1)
            freq_sum[freq_sum == 0] = 1
            self.trans_mat = (self.freq_mat.T/freq_sum).T
            return 1
        except: 
            print("Error has occured")
            return -1 
         
    
    def plotTrans(self, trans_mat = None):
        '''
        plot the transition matrix as a heatmap 
        '''
        if trans_mat is None:
            trans_mat = self.trans_mat
        sns.heatmap(trans_mat, xticklabels= self.states, yticklabels= self.states)
        plt.show()
 