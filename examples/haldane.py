#!/usr/bin/env python

# Version 1.5
# Haldane model from Phys. Rev. Lett. 61, 2015 (1988)

# Copyright under GNU General Public License 2010, 2012
# by Sinisa Coh and David Vanderbilt (see gpl-pytb.txt)

from pytb import * # import TB model class
import numpy as nu
import pylab as pl

# define lattice vectors
lat=[[1.0,0.0],[0.5,nu.sqrt(3.0)/2.0]]
# define coordinates of orbitals
orb=[[1./3.,1./3.],[2./3.,2./3.]]

# make two dimensional tight-binding Haldane model
my_model=tbmodel(2,2,lat,orb)

# set model parameters
delta=0.0
t=-1.0
t2 =0.15*nu.exp((1.j)*nu.pi/2.)
t2c=t2.conjugate()

# set on-site energies
my_model.set_sites([-delta,delta])
# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.add_hop(t, 0, 1, [ 0, 0])
my_model.add_hop(t, 1, 0, [ 1, 0])
my_model.add_hop(t, 1, 0, [ 0, 1])
# add second neighbour complex hoppings
my_model.add_hop(t2 , 0, 0, [ 1, 0])
my_model.add_hop(t2 , 1, 1, [ 1,-1])
my_model.add_hop(t2 , 1, 1, [ 0, 1])
my_model.add_hop(t2c, 1, 1, [ 1, 0])
my_model.add_hop(t2c, 0, 0, [ 1,-1])
my_model.add_hop(t2c, 0, 0, [ 0, 1])

# print tight-binding model
my_model.display()

# generate list of k-points following some high-symmetry line in
# the k-space. Variable kpts here is just an array of k-points
path=[[1.0,0.0],[0.0,1.0]]
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
pl.title("Haldane model band structure")
# make an PDF figure of a plot
pl.savefig("band.pdf")

print
print '---------------------------------------'
print 'starting DOS calculation'
print '---------------------------------------'
print 'Calculating DOS...'

# calculate density of states
# first solve the model on a mesh and return all energies
kmesh=20
kpts=[]
for i in range(kmesh):
    for j in range(kmesh):
        kpts.append([float(i)/float(kmesh),float(j)/float(kmesh)])
# solve the model on this mesh
evals=my_model.solve_all(kpts)
# flatten completely the matrix
evals=evals.flatten()

# plotting DOS
print 'Plotting DOS...'

# now plot density of states
fig=pl.figure()
pl.hist(evals,50,range=(-4.,4.))
pl.ylim(0.0,80.0)
# put title
pl.title("Haldane model density of states")
# make an PDF figure of a plot
pl.savefig("dos.pdf")

print 'Done.\n'
