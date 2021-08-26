#!/usr/bin/env python

# Version 1.5
# Visualization example

# Copyright under GNU General Public License 2010, 2012
# by Sinisa Coh and David Vanderbilt (see gpl-pytb.txt)

from pytb import * # import TB model class
import numpy as nu
import pylab as pl

# define lattice vectors
lat=[[1.0,0.0],[0.5,nu.sqrt(3.0)/2.0]]
# define coordinates of orbitals
orb=[[1./3.,1./3.],[2./3.,2./3.]]

# make two dimensional tight-binding graphene model
my_model=tbmodel(2,2,lat,orb)

# set model parameters
delta=0.0
t=-1.0

# set on-site energies
my_model.set_sites([-delta,delta])
# set hoppings (one for each connected pair of orbitals)
# (amplitude, i, j, [lattice vector to cell containing j])
my_model.add_hop(t, 0, 1, [ 0, 0])
my_model.add_hop(t, 1, 0, [ 1, 0])
my_model.add_hop(t, 1, 0, [ 0, 1])

# visualize infinite model
fig=my_model.visualize(0,1)
fig.savefig("vis_bulk.pdf")

# cutout finite model along direction 0
cut_one=my_model.cut_piece(10,0,glue_edgs=False)
#
fig=cut_one.visualize(0,1)
fig.savefig("vis_ribbon.pdf")

# cutout finite model along direction 1 as well
cut_two=cut_one.cut_piece(10,1,glue_edgs=False)
#
fig=cut_two.visualize(0,1)
fig.savefig("vis_finite.pdf")

print 'Done.\n'
