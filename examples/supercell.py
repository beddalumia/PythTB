#!/usr/bin/env python

# Make arbitrary surface of the graphene model using
# make_supercell method.

# Copyright under GNU General Public License 2010, 2012
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from pythtb import * # import TB model class
import numpy as np
import pylab as pl

# define lattice vectors
lat=[[1.0,0.0],[0.5,np.sqrt(3.0)/2.0]]
# define coordinates of orbitals
orb=[[1./3.,1./3.],[2./3.,2./3.]]

# make two dimensional tight-binding graphene model
my_model=tb_model(2,2,lat,orb)

# set model parameters
delta=0.0
t=-1.0

# set on-site energies
my_model.set_onsite([-delta,delta])
# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.set_hop(t, 0, 1, [ 0, 0])
my_model.set_hop(t, 1, 0, [ 1, 0])
my_model.set_hop(t, 1, 0, [ 0, 1])

# make the supercell of the model
sc_model=my_model.make_supercell([[2,1],[-1,2]],to_home=True)

# now make a slab of the supercell
slab_model=sc_model.cut_piece(6,1,glue_edgs=False)

# visualize slab unit cell
(fig,ax)=slab_model.visualize(0,1)
ax.set_title("Graphene, arbitrary surface")
ax.set_xlabel("x coordinate")
ax.set_ylabel("y coordinate")
fig.savefig("supercell_vis.pdf")

# compute the band structure in the entire band
path=[0.0,1.0]
kpts=k_path(path,100)
evals=slab_model.solve_all(kpts)

# plotting of band structure
print 'Plotting bandstructure...'

# First make a figure object
fig=pl.figure()
# plot all bands
for i in range(evals.shape[0]):
    pl.plot(evals[i],"k-")
# zoom in close to the zero energy
pl.ylim(-1.0,1.0)
# put title on top
pl.title("Graphene arbitrary surface band structure")
pl.xlabel("k-space")
pl.ylabel("Band energy")
# make an PDF figure of a plot
pl.savefig("supercell_band.pdf")

print 'Done.\n'
