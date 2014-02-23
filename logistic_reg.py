from numpy import * 
from scipy.sparse import * 
from scipy.io import * 
from sklearn import linear_model

def TestPath(x_train,y_train,x_test,y_test,l1orl2='l1',C=1.0):
  log_reg = linear_model.LogisticRegression(penalty=l1orl2,C=C)
  log_reg.fit(fmri_train,y_train)
  ans = log_reg.predict(fmri_test)
  
  mistakes = shape(where(ans != y_test.flatten())[0])[1]
  test_error = mistakes/60. 
  print "Test Mistakes", mistakes
  print "Test Error ", test_error
  
  ans = log_reg.predict(fmri_train)
  mistakes = shape(where(ans != y_train.flatten())[0])[1]
  train_error = mistakes/300. 
  print "Tr Mistakes", mistakes
  print "Tr Error ", train_error

  return (1.-train_error),(1.-test_error)

fmri_train = mmread("fmri/subject1_fmri_std.train.mtx")
fmri_test = mmread("fmri/subject1_fmri_std.test.mtx")

y_train = mmread("ytrain_cat.out.mtx")
y_train = y_train.todense()
y_train = y_train.reshape(300,1)

y_test = mmread("ytest_cat.out.mtx")
y_test = y_test.todense()
y_test = y_test.reshape(60,1)

results = zeros((10,3),dtype=float)
idx=0
for i in arange(1.0,0.0,-.1):
  train_error,test_error = TestPath(fmri_train,y_train,fmri_test,y_test,l1orl2='l2',C=i)
  print train_error, test_error
  results[idx][0] = i 
  results[idx][1] = train_error 
  results[idx][2] = test_error 
  #train_error,test_error = TestPath(fmri_train,y_train,fmri_test,y_test,l1orl2='l2')
  idx += 1

print results
