#!/usr/bin/env python

# Version 1.5
# one dimensional chain

# Copyright under GNU General Public License 2010, 2012
# by Sinisa Coh and David Vanderbilt (see gpl-pytb.txt)

from pytb import * # import TB model class
import pylab as pl

# specify model
lat=[[1.0]]
orb=[[0.0]]
my_model=tbmodel(1,1,lat,orb)
my_model.add_hop(-1., 0, 0, [1])

# solve model
path=[[-0.5],[0.5]]
kpts=k_path(path,100)
evals=my_model.solve_all(kpts)

# plot band structure
fig=pl.figure()
pl.plot(evals[0])
pl.title("1D chain band structure")
pl.savefig("band.pdf")
