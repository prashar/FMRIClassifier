import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.io import * 
from scipy.stats import rankdata
from scipy.sparse.sputils import * 

# Does the actual plotting based on coordinates and what to display
def plotbrain(full_vox_coords, voxels_to_display):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = full_vox_coords[:,0]
    ys = full_vox_coords[:,1]
    zs = full_vox_coords[:,2]
    xs = xs[voxels_to_display]
    ys = ys[voxels_to_display]
    zs = zs[voxels_to_display]
    ax.set_xlabel('x voxels')
    ax.set_ylabel('y voxels')
    ax.set_zlabel('z voxels')
    ax.scatter(xs, ys, zs, c='r', marker='o')
    plt.show()

# Does the actual plotting based on coordinates and what to display
def plotsignificantbrain(full_vox_coords, rank_voxels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = full_vox_coords[:,0]
    ys = full_vox_coords[:,1]
    zs = full_vox_coords[:,2]
    ax.set_xlabel('x voxels')
    ax.set_ylabel('y voxels')
    ax.set_zlabel('z voxels')
    ax.scatter(xs, ys, zs, c=rank_voxels, cmap=plt.cm.coolwarm, marker='o')
    plt.show()

def plotfullbrain(coords):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
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
    ax.scatter(xs, ys, zs, c='y', marker='o')
    ax.set_xlabel('x voxels')
    ax.set_ylabel('y voxels')
    ax.set_zlabel('z voxels')
    plt.show()

def plot_imp_voxels_brain(coords):
    importance = np.arange(21764) ; 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = coords[:,0]
    ys = coords[:,1]
    zs = coords[:,2]
    ax.scatter(xs, ys, zs, c=importance,cmap=plt.cm.coolwarm, marker='o')
    ax.set_xlabel('x voxels')
    ax.set_ylabel('y voxels')
    ax.set_zlabel('z voxels')
    plt.show()


# Returns the array containing ranks 
def colmeans(input_data):
    if(isdense(input_data)):
        avg_data = data[:,:].mean(axis=0)
    else:
        avg_data = data.todense()[:,1:].mean(axis=0)
    rank = rankdata(abs(avg_data))
    return rank

# CALC' 'LIPL' 'LT' 'LTRIA' 'LOPER' 'LIPS' 'LDLPFC'};
min_rank = 100 
coords = mmread("coord.out.mtx")
#plot_imp_voxels_brain(coords)
#exit()
#fname = 'correct_weights/new_weight_vec15.out.mtx' 
fname = 'fmri/subject1_fmri_std.train.mtx'
data = mmread(fname)
print np.shape(data)
rank = colmeans(data[275:300,:])
voxel_on_or_off = rank >=(21764-min_rank)
plotsignificantbrain(coords, rank)
