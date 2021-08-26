#!/usr/bin/env python

# Version 1.5
# two dimensional tight-binding checkerboard model

# Copyright under GNU General Public License 2010, 2012
# by Sinisa Coh and David Vanderbilt (see gpl-pytb.txt)

from pytb import * # import TB model class
import numpy as nu
import pylab as pl

# define lattice vectors
lat=[[1.0,0.0],[0.0,1.0]]
# define coordinates of orbitals
orb=[[0.0,0.0],[0.5,0.5]]

# make two dimensional tight-binding checkerboard model
my_model=tbmodel(2,2,lat,orb)

# set model parameters
delta=1.1
t=0.6

# set on-site energies
my_model.set_sites([-delta,delta])
# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.add_hop(t, 1, 0, [0, 0])
my_model.add_hop(t, 1, 0, [1, 0])
my_model.add_hop(t, 1, 0, [0, 1])
my_model.add_hop(t, 1, 0, [1, 1])

# print tight-binding model
my_model.display()

# generate list of k-points following some high-symmetry line in
# the k-space. Variable kpts here is just an array of k-points
path=[[0.0,0.0],[0.0,0.5],[0.5,0.5],[0.0,0.0]]
kpts=k_path(path,100)
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
# put title
pl.title("Checkerboard band structure")
# make an PDF figure of a plot
pl.savefig("band.pdf")

print 'Done.\n'
