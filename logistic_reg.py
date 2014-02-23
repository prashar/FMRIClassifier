from numpy import * 
from scipy.sparse import * 
from scipy.io import * 
from sklearn import linear_model

fmri_train = mmread("fmri/subject1_fmri_std.train.mtx")
fmri_test = mmread("fmri/subject1_fmri_std.test.mtx")

y_train = mmread("ytrain_cat.out.mtx")
y_train = y_train.todense()
y_train = y_train.reshape(300,1)

y_test = mmread("ytest_cat.out.mtx")
y_test = y_test.todense()
y_train = y_train.reshape(300,1)

log_reg = linear_model.LogisticRegression()
log_reg.fit(fmri_train,y_train)
ans = log_reg.predict(fmri_test)

print ans
print y_test

print where(ans != y_test)
mistakes = shape(where(ans != y_test)[0])[1]
print "Mistakes", mistakes
print "Error ", mistakes/60. 


