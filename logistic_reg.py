from numpy import * 
from scipy.sparse import * 
from scipy.io import * 
from sklearn import linear_model

def TestPath(x_train,y_train,x_test,y_test,l1orl2='l1',C=1.0):
  log_reg = linear_model.LogisticRegression(penalty=l1orl2,C=C)
  log_reg.fit(fmri_train,y_train)
  ans = log_reg.predict(fmri_test)
  
  mistakes_test = shape(where(ans != y_test.flatten())[0])[0]
  print "Test Mistakes", mistakes_test
  
  ans = log_reg.predict(fmri_train)
  mistakes_train = shape(where(ans != y_train.flatten())[0])[0]
  print "Tr Mistakes", mistakes_train

  return (mistakes_train),(mistakes_test)

fmri_train = mmread("fmri_train_240.out.mtx")
train_rows = 240
fmri_test = mmread("fmri_train_60.out.mtx")
test_rows = 60

y_train = mmread("ytrain_240_cat.out.mtx")

y_test = mmread("ytrain_60_cat.out.mtx")

idx1 = arange(15,1,-1)
results = zeros((35,3),dtype=int)
idx=0
idx2 = array([1.0,.9,.8,.7,.6,.5,.4,.3,.2,.1,.08,.07,.05,.04,.03,.02,.01,.008,.004,.001],
    dtype=float)
arr = concatenate((idx1,idx2),0)
for i in arr:
  train_error,test_error = TestPath(fmri_train,y_train,fmri_test,y_test,l1orl2='l1',C=1.0/i)
  print train_error, test_error
  results[idx][0] = i 
  results[idx][1] = train_error 
  results[idx][2] = test_error 
  idx += 1

print results
