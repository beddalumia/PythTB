#!/usr/bin/env python

# Version 1.5
# Haldane model from Phys. Rev. Lett. 61, 2015 (1988)
# Calculates density of states for finite sample of Haldane model

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

# print tight-binding model details
my_model.display()

# cutout finite model first along direction x with no PBC
tmp_model=my_model.cut_piece(20,0,glue_edgs=False)
# cutout also along y direction
fin_model_false=tmp_model.cut_piece(20,1,glue_edgs=False)

# cutout finite model first along direction x with PBC
tmp_model=my_model.cut_piece(20,0,glue_edgs=True)
# cutout also along y direction
fin_model_true=tmp_model.cut_piece(20,1,glue_edgs=True)

# solve finite model
evals_true=fin_model_false.solve_all()
evals_true=evals_true.flatten()
evals_false=fin_model_true.solve_all()
evals_false=evals_false.flatten()

# now plot density of states
fig=pl.figure()
pl.hist(evals_true,50,range=(-4.,4.))
pl.ylim(0.0,80.0)
pl.title("Finite Haldane model with PBC")
pl.savefig("dos_true.pdf")  
fig=pl.figure()
pl.hist(evals_false,50,range=(-4.,4.))
pl.ylim(0.0,80.0)
pl.title("Finite Haldane model without PBC")
pl.savefig("dos_false.pdf")  

print 'Done.\n'
