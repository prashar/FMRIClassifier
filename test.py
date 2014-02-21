import unittest
import sys
import shotgunpy
sys.path.append('...')
import numpy as np
from scipy.sparse import *
from scipy.io import *
from time import * 
from numpy import linalg as LA

fmri_train = mmread('fmri/fmri/subject1_fmri_std.train.mtx')
wordid_train = np.loadtxt('fmri/fmri/subject1_wordid.train.mtx')
wordid_train = wordid_train.reshape((300,1))
wfs = mmread('fmri/fmri/word_feature_std.mtx')
print np.shape(fmri_train)
print np.shape(wordid_train)
print wordid_train[0:2]
print np.shape(wfs)


Y_L = [wfs[i-1] for i in wordid_train[:,0]]
Y = np.array(Y_L)
wd_train = wordid_train[:,0]

print "Y_SHAPE" ,  np.shape(Y[:,0])
print "F_TRAIN" ,  np.shape(fmri_train)
final_weight_vector = np.zeros((218,21765))
solver = shotgunpy.ShotgunSolver()
#solver.set_tolerance(.01)
for i in range(0,218):
	print i
	sol = solver.solve_lasso(fmri_train,Y[:,i],25.0)
	final_weight_vector[i,1:] = sol.w
	final_weight_vector[i,0] = sol.offset

print "-------------"
sparse_weight_vector = coo_matrix(final_weight_vector)
mmio.mmwrite('weight_vec25.out',sparse_weight_vector,field='real',precision=25)

# Compute the error rate on training set. 
mistakes = 0 
a = final_weight_vector[:,1:]
w_0 = final_weight_vector[:,0]

for i in range(300):
	b = np.transpose(fmri_train[i])
	sem_vec = a.dot(b) 
	sem_vec += np.transpose(w_0)

        # Calculate the distances to all th 60 words
	L2dist = np.zeros(60)
	idx=0
	for j in wfs:
		dist = LA.norm(sem_vec - j)
		L2dist[idx] = dist
		idx+=1
	indexsmall = np.argmin(L2dist)
	if(indexsmall != (wd_train[i]-1) ):
		print indexsmall,(wd_train[i]-1)
		mistakes += 1
	
print "Mistakes: ", mistakes
