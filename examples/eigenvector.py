#!/usr/bin/env python

# Version 1.5
# prints out eigenvectors of two dimensional tight-binding checkerboard model

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
path=[[0.0,0.0],[0.0,0.5]]
kpts=k_path(path,10)
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
(evals,evects)=my_model.solve_all(kpts,eig_vectors=True)

print 'Print out eigenvalues and eigenvectors'

# go over all k-points
for k in range(len(kpts)):
    print 'k-vector --> ',kpts[k]
      # go over all bands
    for b in range(2):
        print ' band --> ', b
        print ' eigenvalue --> ',evals[b,k]
        # go over all sites
        for s in range(2):
            print '  u_nk(orbital='+str(s)+') --> ',
            print evects[b,k,s]
            
# First make a figure object
fig=pl.figure()
# plot real part of the wavefunction of first band at second orbital
pl.plot(evects[0,:,1].real)
pl.ylim(-1.,1.)
# put title
pl.title("Real part of the wf. of 1st band at 2nd orbital")
# make an PDF figure of a plot
pl.savefig("evec.pdf")

print 'Done.\n'
