from numpy import *
from scipy import *
from numpy import linalg as LA
from matplotlib import pyplot as plt
import scipy.io as io
from scipy.sparse import *
from time import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA

# The whole classifier is a class .. It will be easy to import ..
class FMRIWordClassifier:
    def __init__(self):
        self.fmri_train = None
        self.fmri_test = None
        self.wordid_train = None
        self.wordid_test = None
        self.wfc = None
        self.wfs = None
        self.MAX_RR = 200
        self.sf = 1
        self.precision = .001
        self.lamb = 50

        # Fill up all of the above ..
        self.read_data()

    def setup_reverse_lasso(self,lamb):
        w_init = zeros(218+1)
        X_L = [self.wfs[i-1] for i in self.wordid_train[:,0]]
        X = array(X_L)
        final_weight_vector = zeros((21764,219))
        for i in range(0,21764):
            t1 = time()
            # This call will return 1()X21764 vector
            weight_vector_i = self.solve_lasso(lamb,X,self.fmri_train,300,218,w_init,i)
            final_weight_vector[i,:] = weight_vector_i
            t2 = time()
            print "Iteration {0} took about: ".format(i), t2-t1
        #print shape(final_weight_vector)

        # Write to a file
        # But, first convert to sparse format, ensure the precision and field is set
        sparse_weight_vector = coo_matrix(final_weight_vector)
        io.mmio.mmwrite('weight_vec_rev{0}.out'.format(lamb),sparse_weight_vector,field='real',precision=25)
        #sparse_weight_vector_fromdisk = io.mmio.mmread('weight_vec.out')



    def setup_for_lasso(self,lamb):

        # Setup to Solve Lasso - add the extra term w)
        w_init = zeros(21764+1)

        # The word id numbers range from 1 to 60 - we need to subtract by 1 to get the id from wfs
        # Y_L is actually not going to just be the ID, it'll be the entire 218 vector representing that word
        Y_L = [self.wfs[i-1] for i in self.wordid_train[:,0]]
        Y = array(Y_L)
        #io.mmio.mmwrite('y_out_wfs_train.out',Y,field='real',precision=25)


        # In order to append the 218x21764 vector
        final_weight_vector = zeros((218,21765))
        for i in range(0,218):
            t1 = time()
            # This call will return 1()X21764 vector
            weight_vector_i = self.solve_lasso(lamb,self.fmri_train,Y,240,21764,w_init,i)
            final_weight_vector[i,:] = weight_vector_i
            t2 = time()
            print "Iteration {0} took about: ".format(i), t2-t1
        #print shape(final_weight_vector)

        # Write to a file
        # But, first convert to sparse format, ensure the precision and field is set
        sparse_weight_vector = coo_matrix(final_weight_vector)
        io.mmio.mmwrite('new_weight_vec{0}.out'.format(lamb),sparse_weight_vector,field='real',precision=25)
        #sparse_weight_vector_fromdisk = io.mmio.mmread('weight_vec.out')

    # Read all the data in ..
    def read_data(self):

        # 300 X 21764
        self.fmri_train = io.mmio.mmread("fmri/subject1_fmri_std.train.mtx")
        # 60 X 21764
        self.fmri_test = io.mmio.mmread("fmri/subject1_fmri_std.test.mtx")

        # NOT WORKING - Reading as regular txt
        #self.wordid_train = io.mmio.mmread("fmri/subject1_wordid.train.mtx")
        # 300 X 1
        self.wordid_train = loadtxt("fmri/subject1_wordid.train.mtx",dtype=int)
        self.wordid_train = self.wordid_train.reshape((300,1))
        #self.wordid_train = io.mmio.mmread("fmri/wordid_train_240.out.mtx")

        #self.wordid_test = io.mmio.mmread("fmri/wordid_train_60.out.mtx")
        #self.wordid_test = io.mmio.mmread("fmri/wordid_60_2cand.out.mtx")


        # 60 X 2
        self.wordid_test = io.mmio.mmread("fmri/subject1_wordid.test.mtx")

        # 60 X 218
        self.wfc = io.mmio.mmread("fmri/word_feature_centered.mtx")

        # 60 X 218
        self.wfs = io.mmio.mmread("fmri/word_feature_std.mtx")

    # What we're trying to do here is to get a 21764 weight vector for each
    # semantic feature. There are 218 features, so we'll have a large 218 by
    # 21764 matrix somewhere. To calculate the weight of a
    # perspective Yij where i is the word id and j is the semantic feature, we
    # will take the dot product of that 21764 weight vector with the input
    # provided to indicate the score for that feature. The final output will be
    # a 218 element array each filled with the score for the respective feature.

    def solve_lasso(self,lamb,X,Y,N,d,init_w,y_idx):

        # Initialize the w to be 0
        w_tilde = copy(init_w) ;
        w_prev = zeros(d+1)

        # Set round robin to be 0
        round_robin = 0

        # Set w_0 to be zero
        w_0 = init_w[0]

        # Precompute Xij^2
        X_squared = X**2
        a_j_pre = X_squared.sum(axis=0) * 2
        new_error = 0

        Xw = X.dot(transpose(w_tilde[1:]))
        while(round_robin < self.MAX_RR):
            # Copy the entire vector coming in
            w_prev = copy(w_tilde)

            for voxel_j in range(0,d):
                c_j = 2 * X[:,voxel_j].dot(Y[:,y_idx] -Xw + (X[:,voxel_j] * w_tilde[voxel_j+1]) - w_0)
                w_tilde_j_old = w_tilde[voxel_j+1]
                if(c_j < (-1 * lamb)):
                    w_tilde[voxel_j+1] = (c_j + lamb)/a_j_pre[voxel_j]
                elif(c_j > lamb):
                    w_tilde[voxel_j+1] = (c_j - lamb)/a_j_pre[voxel_j]
                else:
                    w_tilde[voxel_j+1] = 0
                delta_w = w_tilde[voxel_j+1] - w_tilde_j_old
                Xw += delta_w * X[:,voxel_j]

            #Recalculate w_0
            w_0 = sum(Y[:,y_idx] - X.dot(w_tilde[1:]))/N
            w_tilde[0] = w_0
            diff_w = w_tilde - w_prev
            new_error = max(abs(diff_w))
            #print new_error
            round_robin += 1
            if new_error < self.precision:
                print("Done in ", round_robin)
                return w_tilde

        print("Not done after MAX Iterations - Exiting!")
        return w_tilde

    def SolveFastLasso(self,trainOrTest=0,dist=0):
        # Train
        Y_L = [self.wfs[i-1] for i in self.wordid_train[:,0]]
        Y = array(Y_L)
        lasso = linear_model.Lasso(alpha=25.0)
        lasso.fit(self.fmri_train,Y)

        w_0 = lasso.intercept_
        a = lasso.coef_
        if(trainOrTest == 1):
            Y_sol = self.wordid_test[:,0]
        else:
            Y_sol = self.wordid_train[:,0]

        mistakes = 0
        # Iterate over the total number of 60 words to see
        # which word gives us the smallest L2 Dist from our
        # computed semantic vector. The semantic vector
        # gets computed within the loop/
        if(trainOrTest == 1):
            CompVector = self.fmri_test
        else:
            CompVector = self.fmri_train

        for i in range(shape(CompVector)[0]):

            # Test against one row at a time.
            b = CompVector[i]

            # Compute the semantic vector
            semantic_vec = a.dot(b)
            #print shape(semantic_vec)

            semantic_vec += w_0

            # Compare against all candidate words
            if(self.CalculateL2DistAgainstTestSet(semantic_vec,Y_sol,i,dist)):
                mistakes += 1
        print "Total Mistakes: ", mistakes

    def CalcSemanticFeatureVector(self,trainOrTest=1,fread='weight_vec0.out.mtx',dist=0):
        # Read the input weight_vector
        weight_vector = io.mmio.mmread(fread)
        weight_vector_dense = weight_vector.todense()
        a = weight_vector_dense[:,1:]
        '''
        b = array(weight_vector_dense[0,1:])
        print shape(b)
        print "Nonzero elements", nonzero(b)[1]
        cnt = shape(nonzero(b))[1]
        arr = zeros(cnt)
        for i in range(cnt):
            arr[i] = b[nonzero(b)[0][i], nonzero(b)[1][i]]
        print arr
        '''
        # Need to add w_0 to the final answer
        w_0 = weight_vector_dense[:,0]
        #print w_0
        #return

        # Get the list of candidate words - We'll pass this in to to the function
        # that does L2 distance computation.
        # 1 for trainOrTest implies TEST
        if(trainOrTest == 1):
            Y = self.wordid_test
        else:
            Y = self.wordid_train[:,0]

        mistakes = 0
        #print w_0

        # Iterate over the total number of 60 words to see
        # which word gives us the smallest L2 Dist from our
        # computed semantic vector. The semantic vector
        # gets computed within the loop/
        if(trainOrTest == 1):
            CompVector = self.fmri_test
        else:
            CompVector = self.fmri_train

        for i in range(shape(CompVector)[0]):

            # Test against one row at a time.
            b = transpose(CompVector[i])

            # Compute the semantic vector
            semantic_vec = a.dot(b)
            #print shape(semantic_vec)

            semantic_vec += transpose(w_0)

            #candidate = self.wordid_test[i, 1]

            # Compare against all candidate words
            if(self.CalculateL2DistAgainstTestSet(semantic_vec,Y,i,dist)):
                mistakes += 1
        print "Total Mistakes: ", mistakes

    # This routine measures the l2 or cosine difference between
    # the semantic vector and all 60 words, and tries to guess a match
    # based on the user preference for dist metric.
    # dist = 0 is eucld, 1 is cosine
    def CalculateL2DistAgainstTestSet(self, semantic_vec, Y, expected_idx, dist_metric=0, candidate=0):

        # Will hold the distance metric idx
        compIndex = 0

        l2dist = euclidean_distances(semantic_vec,self.wfs)
        l2idx = argmin(l2dist)

        truth_wordid = Y[expected_idx][0] - 1
        candidate_wordid = Y[expected_idx][1] - 1

        wfsfortest = zeros((2, 218), dtype=float)

        wfsfortest[0] = self.wfs[truth_wordid]
        wfsfortest[1] = self.wfs[candidate_wordid]

        # Also compute the cosine similarity matrix - this one seems to give better results.
        cossim = cosine_similarity(semantic_vec, wfsfortest)
        #cossim2 = cosine_similarity(semantic_vec,self.wfs[candidate_wordid])
        #cosidx = -1
        cosidx = argmax(cossim)

        if (cosidx == 0):
            cosidx = truth_wordid
        else:
            cosidx = candidate_wordid

        if(dist_metric == 0):
            # L2 Distances
            compIndex = l2idx
        else:
            compIndex = cosidx

        print "Expected Word Index: ", truth_wordid, " | CosIDX: ", cosidx, " | EucIdx", l2idx
        if(compIndex != truth_wordid):
            return True

        # No Mistake
        return False

    def PlotSVD(self):
        U,S,V = linalg.svd(self.fmri_train,full_matrices=1)
        plt.plot(S)
        plt.show()

    def PlotPCA(self):
        pca = RandomizedPCA(n_components=21764)
        print shape(self.fmri_train)
        pca.fit(self.fmri_train)
        plt.plot(pca.explained_variance_ratio_)
        plt.show()
        print pca.get_params()

def main():
    classifier = FMRIWordClassifier()
    '''
    for i in [30,35,40,0]:
        classifier.setup_for_lasso(i)
        print "Done with lambda ", i
    '''
    # Run for lambda == 0
    # This will store the a file named weight_vector0.out.mtx on your disk containing the resulting matrix
    #classifier.setup_for_lasso(5)
    #classifier.setup_reverse_lasso(25)


    #classifier.SolveFastLasso(trainOrTest=0,dist=0)

    # 0 to test on training samples(300), #1 to test on testing samples
    # dist == 0 for l2, dist == 1 for cosine
    #classifier.CalcSemanticFeatureVector(trainOrTest=0,fread='new_weight_vec70.out.mtx',dist=1)
    #classifier.CalcSemanticFeatureVector(trainOrTest=0,fread='new_Fweight_vec10.out.mtx',dist=1)
    classifier.PlotPCA()

'''
    X = array([[0,0],[1,1],[2,2]])
    Y = array([[0],[1],[2]])
    N = 3
    d = 2
    init_w = zeros(3)
    y_idx=0
    ret = classifier.solve_lasso(0.1,X,Y,N,d,init_w,y_idx)
    print ret
'''
if __name__ == '__main__':
    main()
