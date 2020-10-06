
"""
Created on Fri Jul 10 09:56:23 2020

@author: danielstilckfranca
"""



#code used for "Limitations of optimization algorithms on noisy quantum devices"
#it includes functions to generate random instances of adjacency matrices of regular graphs
#the SK model and a suboptimal implementation of the Metropolis algorithm, besides a 
#function to compute the partition function by brute force. 
import itertools
import numpy as np
import random
from scipy.linalg import expm, sinm, cosm
import pandas as pd







def SK_model(n):
#generates a random instance of the sk model with n spins.
    A=np.zeros([n,n])

    for k in range(0,n):
        for j in range(0,n):
            if j is not k:
                blob=(1/np.sqrt(n))*np.random.normal(0,1)
                A[k,j]=blob/2
                A[j,k]=blob/2
    return A

def random_graph(n,delta):
    #generates a random adjacency matrix of a delta-regular graph with n vertices
    A=np.zeros([n,n])
    for k in range(0,n):
        for neigh in range(0,delta):
            y=random.randint(0,n-1)
            if y is not k:
                A[k,y]=1
                A[y,k]=1
    return A
            
def update(current,beta,A):
    #one step of the update of the metropolis algorithm  given input state current
    # and inverse temperature beta with Hamiltonian given by A
    n=A.shape[0]

    x=random.randint(0,n)-1
    proposal=current.copy()
    proposal[x]=proposal[x]*(-1)

    proposal_trans=np.transpose(proposal)
    current_trans=np.transpose(current)
    new_energy=proposal_trans.dot(A)
    new_energy=new_energy.dot(proposal)
    old_energy=current_trans.dot(A)
    old_energy=old_energy.dot(current)
    
    if new_energy<old_energy:
        #print("acccepted smaller",new_energy,old_energy)
        current=proposal
    else:
        p=random.random()
        if np.exp(-beta*(new_energy-old_energy))>p:
            current=proposal
            #print("acccepted larger",new_energy,old_energy)
        
    
    return current


 



def metropolis(beta,A,times,current):
    #runs the Metropolis algorithm for inveerse temperature beta, Hamiltonian A, times time steps
    #and initial sttate given by current.
    n=A.shape[0]
    
    for k in range(0,times):
        current=update(current,beta,A)
    

    return current




def partition(A,beta):
    #computes the log-partition function by brute force at inverse temperature beta
    #and Hamiltonian A. Alrerady returns divided by 2**n and with a minus sign

    n=A.shape[0]
    estimate=0
    for initial in itertools.product([-1, 1], repeat=n):
        initial=np.array(initial)
        new_energy=initial.dot(A)
        new_energy=initial.dot(new_energy)
       
        estimate=estimate+np.exp(-beta*new_energy)
     
            
    end = time.time()

    return -np.log(estimate/(2**n))

def ratio_est(beta1,beta2,H,sampler_func,samples,steps_sampler):
    #assume going from beta1 to beta2
    results=[]
    for k in range(0,samples):
        new_sample=sampler_func(beta1,H,steps_sampler,np.ones([H.shape[0]]))
        energy=new_sample.dot(H)
        energy=energy.dot(new_sample)

        results.append(np.exp(energy*(beta2-beta1)))
    return sum(results)/samples


def esti_partition(annealing_schedule,H,sampler_func,samples,steps_sampler):
    #estimates the partition function going through an annealing schedule specified by
    #annealing schedule
    results=[]
    for k in range(0,len(annealing_schedule)-1):
        beta1=annealing_schedule[k]
        beta2=annealing_schedule[k+1]
        results.append(ratio_est(beta1,beta2,H,sampler_func,samples,steps_sampler))
    log_results=np.log(results)
    return log_results
        



    

