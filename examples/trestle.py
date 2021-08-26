#!/usr/bin/env python

# one dimensional tight-binding model of a trestle-like structure

# Copyright under GNU General Public License 2010, 2012
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from pythtb import * # import TB model class
import numpy as np
import pylab as pl

# define lattice vectors
lat=[[2.0,0.0],[0.0,1.0]]
# define coordinates of orbitals
orb=[[0.0,0.0],[0.5,1.0]]

# make one dimensional tight-binding model of a trestle-like structure
my_model=tb_model(1,2,lat,orb,per=[0])

# set model parameters
t_first=0.8+0.6j
t_second=2.0

# leave on-site energies to default zero values
# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.set_hop(t_second, 0, 0, [1,0])
my_model.set_hop(t_second, 1, 1, [1,0])
my_model.set_hop(t_first, 0, 1, [0,0])
my_model.set_hop(t_first, 1, 0, [1,0])

# print tight-binding model
my_model.display()

# generate list of k-points following some high-symmetry line in
kpts=k_path('full',100)

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
pl.title("Trestle band structure")
pl.xlabel("Path in k-space")
pl.ylabel("Band energy")
# make an PDF figure of a plot
pl.savefig("trestle_band.pdf")

print 'Done.\n'
