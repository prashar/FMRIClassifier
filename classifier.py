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
    # Fill up all of the above .. 
    self.read_data() 
  
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

def main():
  classifier = FMRIWordClassifier()

if __name__ == '__main__':
  main()
