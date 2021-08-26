#!/usr/bin/env python

# zero dimensional tight-binding model of a NH3 molecule

# Copyright under GNU General Public License 2010, 2012
# by Sinisa Coh and David Vanderbilt (see gpl-pythtb.txt)

from pythtb import * # import TB model class
import numpy as np
import pylab as pl

# define lattice vectors
lat=[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
# define coordinates of orbitals
sq32=np.sqrt(3.0)/2.0
orb=[[ (2./3.)*sq32, 0.   ,0.],
     [(-1./3.)*sq32, 1./2.,0.],
     [(-1./3.)*sq32,-1./2.,0.],
     [  0.         , 0.   ,1.]]
# make zero dimensional tight-binding model
my_model=tb_model(0,3,lat,orb)

# set model parameters
delta=0.5
t_first=1.0

# change on-site energies so that N and H don't have the same energy
my_model.set_onsite([-delta,-delta,-delta,delta])
# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j)
my_model.set_hop(t_first, 0, 1)
my_model.set_hop(t_first, 0, 2)
my_model.set_hop(t_first, 0, 3)
my_model.set_hop(t_first, 1, 2)
my_model.set_hop(t_first, 1, 3)
my_model.set_hop(t_first, 2, 3)

# print tight-binding model
my_model.display()

print '---------------------------------------'
print 'starting calculation'
print '---------------------------------------'
print 'Calculating bands...'
print
print 'Band energies'
print    
# solve for eigenenergies of hamiltonian
evals=my_model.solve_all()

# First make a figure object
fig=pl.figure()
# plot all states
pl.plot(evals,"bo")
pl.ylim(-1.8,3.2)
pl.xlim(-1.,4.)
# put title
pl.title("Molecule levels")
pl.xlabel("Orbital")
pl.ylabel("Energy")
# make an PDF figure of a plot
pl.savefig("0dim_spectrum.pdf")                    

print 'Done.\n'
