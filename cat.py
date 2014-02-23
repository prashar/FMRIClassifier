from numpy import * 
from scipy.io import * 
from scipy.sparse import *



#idx = mmio.mmread('ytrain_cat.out')
#print shape(idx.todense())
#idx.todense()[0]


categories = [['bear','cat','cow','dog','horse'],['arm','eye','foot','hand','leg'],['apartment','barn','church','house','igloo'],
        ['arch','chimney','closet','door','window'],['coat','dress','pants','shirt','skirt'],
        ['bed','chair','desk','dresser','table'],['ant','bee','beetle','butterfly','fly'],
        ['bottle','cup','glass','knife','spoon'],['bell','key','refrigerator','telephone','watch'],
        ['chisel','hammer','pliers','saw','screwdriver'],
        ['carrot','celery','corn','lettuce','tomato'],['airplane','bicycle','car','train','truck']]

words = genfromtxt("fmri/meta/dictionary.txt",dtype='str')
print shape(words)
cat_words = zeros(60)
wid_cats = zeros(60)

wordid_train = loadtxt("fmri/subject1_wordid.train.mtx",dtype=int)
wordid_train = wordid_train.reshape((300,1))
wordid_test = mmread("fmri/subject1_wordid.test.mtx")


# MAP ALL THE WORD IDS TO THEIR CATS ..
idx_word = 0
for word in words:
    idx_cat = 0
    for cat in categories: 
        if word in cat:
            cat_words[idx_word] = idx_cat
        idx_cat += 1
    idx_word += 1

idx_wid = 0 
for wid in wordid_test[:,0]:
    wid_cats[idx_wid] = cat_words[wid-1]
    idx_wid += 1
'''
for i in range(300):
    print "ID,CAT,WID_CAT", wordid_train[i], wid_cats[i]
'''
print cat_words
print words
print wid_cats

weight_vector = coo_matrix(wid_cats,(60,1))
mmio.mmwrite('ytest_cat.out',weight_vector,field='integer')

