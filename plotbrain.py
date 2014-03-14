import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.io import * 
from scipy.stats import rankdata

def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 10

coords = mmread("coord.out.mtx")
# CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'};

'''

#for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
for c, m, zl, zh in [('r', 'o', 0, 50)]:
    xs = randrange(n, 0, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zl, zh)
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

print "coordinates:"
print np.hstack((xs.reshape(n,1),ys.reshape(n,1),zs.reshape(n,1)))

print "individ"
print xs.flatten()
print ys.flatten()
print zs.flatten()

'''

fname = 'weight_vec25.out.mtx' 
data = mmread(fname)
print np.shape(data)
avg_data = data.todense()[:,1:].mean(axis=0)

# Random 1 X 21764 array
min_rank = 10 
rank = rankdata(abs(avg_data))
xs = coords[:,0]
ys = coords[:,1]
zs = coords[:,2]
ax.set_xlabel('x voxels')
xs = xs[rank>=(21764-min_rank)]
ys = ys[rank>=(21764-min_rank)]
ax.set_ylabel('y voxels')
zs = zs[rank>=(21764-min_rank)]
ax.set_zlabel('z voxels')


'''
xs = coords[:5441,0]
ys = coords[:5441,1]
zs = coords[:5441,2]
ax.scatter(xs, ys, zs, c='r', marker='o')
xs = coords[5441:10882,0]
ys = coords[5441:10882,1]
zs = coords[5441:10882,2]
ax.scatter(xs, ys, zs, c='b', marker='o')
xs = coords[10882:16323,0]
ys = coords[10882:16323,1]
zs = coords[10882:16323,2]
ax.scatter(xs, ys, zs, c='g', marker='o')
xs = coords[16323:21764,0]
ys = coords[16323:21764,1]
zs = coords[16323:21764,2]
'''
ax.scatter(xs, ys, zs, c='r', marker='o')
plt.show()

