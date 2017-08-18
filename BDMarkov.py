import numpy as np
import pydot
from IPython.display import Image
from trainMarkov import MarkovChain

class BirthDeathChain(MarkovChain): # () 
    """
    Approximate the markov chain with a birth and death markov chain such that will minimize the transition probability 
    to non-neighboring states 
    """
    def __init__(self):
        super(BirthDeathChain, self).__init__()
        # ---- these attributes will be set after calling fit method
        self.BDtrans_mat = None
        self.BD_states = []
        self.BDfreq_mat = None
        self.BDfreq_mat_aprox = None
        self.BDtrans_mat_aprox = None
        self.aprox_cost = None
        
    def getTransMat(self, freq_mat = None):
        """
        get the transition matrix from the frequency matrix, if freq_mat is null the object's transition matrix is 
        obtained from the object's frequency matrix
        """        
        change_object_mat = False
        if freq_mat is None:
            change_object_mat = True
            freq_mat = self.freq_mat
        freq_sum = np.sum(freq_mat, axis = 1)
        freq_sum[freq_sum == 0] = 1
        if change_object_mat:
            self.trans_mat = (freq_mat.T/freq_sum).T
        else: return (freq_mat.T/freq_sum).T
        
        
    def triGain(self, order = None):
        """
        Diagonal gain function for changing the order of the states, 
        if order is null gives the sum of tri diagonal frequency
        """
        freq_mat_copy = self.freq_mat.copy()

        if not order:
            freq_mat_copy = freq_mat_copy[order][:, order]

        new_gain = sum(np.diagonal(freq_mat_copy,-1))+ \
                   sum(np.diagonal(freq_mat_copy,0))+ \
                   sum(np.diagonal(freq_mat_copy,1))

        prev_gain = sum(np.diagonal(freq_mat_copy,-1))+ \
                    sum(np.diagonal(freq_mat_copy,0))+ \
                    sum(np.diagonal(freq_mat_copy,1))
        if not order:
            return prev_gain

        return new_gain - prev, prev_gain, new_gain        
   

    def replaceTriDiag(self, matrix, replace_portion = None):
        """
        Change triDiag matrix 
        Parameters: 
        matrix: input matrix
        replace_portion: default is 'tri_diag', replaces the tridiagonal values of the matrix with zeros 
                         when set to 'non_tri_diag', replaces the values of non tri-diagonal elements with zeros 
        """
        num_states = len(matrix)
        diag_indx = np.arange(num_states-1)

        if(replace_portion == 'tri_diag' or replace_portion is None):
            mat_copy = matrix.copy()
            mat_copy[diag_indx, diag_indx+1] = 0
            mat_copy[diag_indx, diag_indx] = 0
            mat_copy[-1, -1] = 0 
            mat_copy[diag_indx+1, diag_indx] = 0
            
        if(replace_portion == 'non_tri_diag'):
            mat_copy = np.zeros([num_states,num_states])
            mat_copy[diag_indx, diag_indx+1] = matrix[diag_indx, diag_indx+1]
            mat_copy[diag_indx, diag_indx] = matrix[diag_indx, diag_indx]
            mat_copy[-1, -1] = matrix[-1, -1]
            mat_copy[diag_indx+1, diag_indx] = matrix[diag_indx+1, diag_indx]

        return mat_copy

  
    def transCostFun(self, freq_mat = None):       
        """
        Transition probability cost function
        """
        if freq_mat is None:
            freq_mat = self.freq_mat.copy()
            
        freq_mat_copy = freq_mat.copy()
        
        freq_mat_copy = self.replaceTriDiag(freq_mat_copy)
        freq_sum = np.sum(freq_mat, axis = 1)
        freq_sum[freq_sum == 0] = 1
        # nansum to treat those values with zero rows have zero weight 
        # gives the average cost (percentage of transition)
        return sum(np.sum(freq_mat_copy, axis = 1)/freq_sum)/len(self.states) 
    
    
    def fit(self, seq_data, err_thr= None, max_iter = None):
        try: 
            super(BirthDeathChain, self).fit(seq_data)
            ''' 
            Find the birth-death process to minimize transitioning to non-neighboring states
            Dependencies numpy 
            Default values for optimization parameters'''

            if max_iter is None: 
                max_iter = 10000
            if err_thr is None: 
                err_thr = 0.01

            order = np.arange(0,len(self.states))

            mat_copy = self.freq_mat.copy()
            prev_order = order.copy()
            opt_order = prev_order 
            prev_cost = self.transCostFun()
            opt_cost = np.copy(prev_cost)
            cost_arr= [prev_cost]
            i=0
            while i < max_iter and opt_cost > err_thr: 
                np.random.shuffle(prev_order)
                new_order = prev_order.copy()
                mat_new = mat_copy[new_order][:, new_order]
                prev_cost = self.transCostFun(mat_new)
                if prev_cost < opt_cost:
                    opt_cost = prev_cost 
                    opt_order = new_order.copy()
                cost_arr.append(prev_cost)    
                i+=1


                self.BDfreq_mat = self.freq_mat[opt_order][:, opt_order]
                self.BDtrans_mat = self.trans_mat[opt_order][:, opt_order]
                self.BD_states = [self.states[state_indx] for state_indx in opt_order]

                self.BDfreq_mat_aprox = self.replaceTriDiag(self.BDfreq_mat, replace_portion ='non_tri_diag')
                self.BDtrans_mat_aprox = self.getTransMat(self.BDfreq_mat_aprox)
                self.aprox_cost = opt_cost
            return cost_arr
        except: 
            print("Error has occured")
            return -1                 

            
    def graphBDChain(self, state_color = None, file_name = None, file_nameBD = None):
        """
        pydot 
        dependencies pydot and graphviz 
        """
        if file_name is None:
            file_name = 'Connected_graph.png'
        if file_nameBD is None: 
            file_nameBD = 'BD_graph.png'      
        if state_color is None:
            state_color="#976856"
            
        # specify a directed graph
        graph = pydot.Dot(graph_type='digraph', center=True, concentrate = False, dimen= 3)
        graphBD = pydot.Dot(graph_type='digraph', center=True, concentrate = False, dimen= 3)
        for state_i in self.BD_states: 
            node_i = pydot.Node(state_i, style="filled", fillcolor=state_color)
            graph.add_node(node_i)
            graphBD.add_node(node_i)

        for i in range(len(self.BD_states)): 
            state_i = self.BD_states[i]
            for j in range(len(self.BD_states)):
                state_j = self.BD_states[j]
                if self.BDtrans_mat[i,j]>0:
                    if abs(j-i)<=1: 
                        edge = pydot.Edge(state_i,state_j,  label=str(round(self.BDtrans_mat[i,j],2)))
                        edgeBD = pydot.Edge(state_i,state_j,  label=str(round(self.BDtrans_mat_aprox[i,j],2)))
                        graphBD.add_edge(edgeBD)
                    else: edge = pydot.Edge(state_i,state_j, 
                                            label=str(round(self.BDtrans_mat[i,j],2)),
                                            color="blue", style="dotted")
                    graph.add_edge(edge)

        graph.write_png(file_name)
        graphBD.write_png(file_nameBD)
        Image(filename=file_name) 
        Image(filename=file_nameBD)
        return
    
   