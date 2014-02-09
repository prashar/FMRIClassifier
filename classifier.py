from numpy import *
from scipy import *
from numpy import linalg as LA
from matplotlib import pyplot as plt
import scipy.io as io

# The whole classifier is a class .. It will be easy to import ..
class FMRIWordClassifier:
    def __init__(self):
        self.fmri_train = None
        self.fmri_test = None
        self.wordid_train = None
        self.wordid_test = None
        self.wfc = None
        self.wfs = None
        self.MAX_RR = 50
        self.sf = 1
        self.precision = .001
        self.lamb = 30

        # Fill up all of the above ..
        self.read_data()

    def setup_for_lasso(self):
        # Setup to Solve Lasso
        w_init = zeros(21764+1)
        Y_L = [self.wfs[i-1] for i in self.wordid_train[:,0]]
        Y = array(Y_L)
        # In order to append the 218x21764 vector
        old = None
        for i in range(0,2):
            ans = self.solve_lasso(self.lamb,self.fmri_train,Y,300,21764,w_init)
            print(shape(ans))
            if i>0:
                ans = append([old],[ans],axis=0)
            old = ans
        #ans = self.solve_lasso_insane(self.lamb,self.fmri_train,Y,300,21764,w_init)
        print shape(ans)

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

    def solve_lasso(self,lamb,X,Y,N,d,init_w):

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
                c_j = 2 * X[:,voxel_j].dot(Y[:,0] -Xw + (X[:,voxel_j] * w_tilde[voxel_j+1]) - w_0)
                w_tilde_j_old = w_tilde[voxel_j+1]
                if(c_j < (-1 * lamb)):
                    w_tilde[voxel_j+1] = (c_j + lamb)/a_j_pre[voxel_j]
                elif(c_j > lamb):
                    w_tilde[voxel_j+1] = (c_j - lamb)/a_j_pre[voxel_j]
                else:
                    w_tilde[voxel_j+1] = 0
                delta_w = w_tilde[voxel_j+1] - w_tilde_j_old
                Xw += delta_w * X[:,voxel_j]
            w_0 = sum(Y[:,0] - X.dot(w_tilde[1:]))/N
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

def main():
    classifier = FMRIWordClassifier()
    classifier.setup_for_lasso()

if __name__ == '__main__':
    main()
