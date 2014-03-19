from numpy import *
from scipy.io import *
from scipy import ndimage

# Write the gaussian smoothed data
fmri_all = mmread("fmri/subject1_fmri_std.train.mtx")
print shape(fmri_all)
out_data = zeros((300,21764))
for i in range(shape(fmri_all)[0]):
    print 'input shape ', shape(fmri_all[i])
    g_out = ndimage.gaussian_filter(fmri_all[i],sigma=5)
    print 'output shape ', shape(g_out)
    out_data[i] = g_out
mmwrite('smooth_vector_300_21764.out',out_data,field='real',precision=25)


