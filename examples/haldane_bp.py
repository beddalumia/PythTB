#!/usr/bin/env python

# Version 1.5
# Haldane model from Phys. Rev. Lett. 61, 2015 (1988)
# Calculates Berry phases and curvatures for this model

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

print r"Using approach #1"
# approach #1
# generate object of type wf_array that will be used for
# Berry phase and curvature calculations
my_array_1=wf_array(my_model,[31,31])
# solve model on a regular grid, and put origin of
# Brillouin zone at -1/2 -1/2 point
my_array_1.solve_on_grid([-0.5,-0.5])
# calculate Berry phases along direction k_x for lower band
phi_a_1 = my_array_1.berry_phase([0],0,contin=True)
# calculate Berry phases along direction k_x for upper band
phi_b_1 = my_array_1.berry_phase([1],0,contin=True)
# calculate Berry phases along direction k_x for both bands
phi_c_1 = my_array_1.berry_phase([0,1],0,contin=True)
# calculate Berry curvature for lower band
curv_a_1=my_array_1.berry_curv([0])

# plot Berry phases
fig=pl.figure()
pl.plot(phi_a_1, 'o')
pl.plot(phi_b_1, 'o')
pl.plot(phi_c_1, 'o')
fig.savefig("phase.pdf")
# print out info about curvature
print " Berry curvature= ",curv_a_1

print r"Using approach #2"
# approach #2
# do the same thing as in approach #1 but do not use
# automated solver
#
# intialize k-space mesh
nkx=31
nky=31
kx=nu.linspace(-0.5,0.5,num=nkx)
ky=nu.linspace(-0.5,0.5,num=nky)
# initialize object to store all wavefunctions
my_array_2=wf_array(my_model,[nkx,nky])
# solve model at all k-points
for i in range(nkx):
    for j in range(nky):
        (eval,evec)=my_model.solve_one([kx[i],ky[j]],eig_vectors=True)
        # store wavefunctions
        my_array_2.add_wf([i,j],evec)
# impose periodic boundary conditions in both k_x and k_y directions
my_array_2.impose_pbc([True,True])
# calculate Berry curvature for lower band
curv_a_2=my_array_2.berry_curv([0])

# print out info about curvature
print " Berry curvature= ",curv_a_2

print 'Done.\n'
