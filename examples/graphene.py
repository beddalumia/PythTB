#!/usr/bin/env python

# Toy graphene model

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

# print tight-binding model
my_model.display()

# generate list of k-points following some high-symmetry line in
# the k-space. Variable kpts here is just an array of k-points
path=[[1.0,0.0],[0.0,1.0]]
kpts=k_path(path,200)
print '---------------------------------------'
print 'report of k-point path'
print '---------------------------------------'
print 'Path runs over',len(kpts),'k-points connecting:'
for k in path:
    print k
print
    
print '---------------------------------------'
print 'starting calculation'
print '---------------------------------------'
print 'Calculating bands...'

# solve for eigenenergies of hamiltonian on
# the set of k-points from above
evals=my_model.solve_all(kpts)

# plotting of band structure
print 'Plotting bandstructure...'

# First make a figure object
fig=pl.figure()
# plot first band
pl.plot(evals[0])
# plot second band
pl.plot(evals[1])
# put title on top
pl.title("Graphene band structure")
pl.xlabel("Path in k-space")
pl.ylabel("Band energy")
# make an PDF figure of a plot
pl.savefig("graphene_band.pdf")

print 'Done.\n'
