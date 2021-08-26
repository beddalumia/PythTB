#!/usr/bin/env python

# Two dimensional tight-binding 2D Kane-Mele model
# C.L. Kane and E.J. Mele, PRL 95, 146802 (2005) Eq. (1)

# Copyright under GNU General Public License 2010, 2012
# by Sinisa Coh and David Vanderbilt (see gpl-pytb.txt)

from pythtb import * # import TB model class
import numpy as np
import pylab as pl

def get_kane_mele(topological):
  "Return a Kane-Mele model in the normal or topological phase."

  # define lattice vectors
  lat=[[1.0,0.0],[0.5,np.sqrt(3.0)/2.0]]
  # define coordinates of orbitals
  orb=[[1./3.,1./3.],[2./3.,2./3.]]
  
  # make two dimensional tight-binding Kane-Mele model
  ret_model=tbmodel(2,2,lat,orb,nspin=2)
  
  # set model parameters depending on whether you are in the topological
  # phase or not
  if topological=="even":
    esite=2.5
  elif topological=="odd":
    esite=1.0
  # set other parameters of the model
  thop=1.0
  spin_orb=0.6*thop*0.5
  rashba=0.25*thop
  
  # set on-site energies
  ret_model.set_sites([esite,(-1.0)*esite])
  
  # set hoppings (one for each connected pair of orbitals)
  # (amplitude, i, j, [lattice vector to cell containing j])
  
  # useful definitions
  sigma_x=np.array([0.,1.,0.,0])
  sigma_y=np.array([0.,0.,1.,0])
  sigma_z=np.array([0.,0.,0.,1])
  
  # spin-independent first-neighbor hoppings
  ret_model.set_hop(thop, 0, 1, [ 0, 0])
  ret_model.set_hop(thop, 0, 1, [ 0,-1])
  ret_model.set_hop(thop, 0, 1, [-1, 0])
  
  # second-neighbour spin-orbit hoppings (s_z)
  ret_model.set_hop(-1.j*spin_orb*sigma_z, 0, 0, [ 0, 1])
  ret_model.set_hop( 1.j*spin_orb*sigma_z, 0, 0, [ 1, 0])
  ret_model.set_hop(-1.j*spin_orb*sigma_z, 0, 0, [ 1,-1])
  ret_model.set_hop( 1.j*spin_orb*sigma_z, 1, 1, [ 0, 1])
  ret_model.set_hop(-1.j*spin_orb*sigma_z, 1, 1, [ 1, 0])
  ret_model.set_hop( 1.j*spin_orb*sigma_z, 1, 1, [ 1,-1])
  
  # Rashba first-neighbor hoppings: (s_x)(dy)-(s_y)(d_x)
  r3h =np.sqrt(3.0)/2.0
  # bond unit vectors are (r3h,half) then (0,-1) then (-r3h,half)
  ret_model.set_hop(1.j*rashba*( 0.5*sigma_x-r3h*sigma_y), 0, 1, [ 0, 0], mode="add")
  ret_model.set_hop(1.j*rashba*(-1.0*sigma_x            ), 0, 1, [ 0,-1], mode="add")
  ret_model.set_hop(1.j*rashba*( 0.5*sigma_x+r3h*sigma_y), 0, 1, [-1, 0], mode="add")

  return ret_model

# now solve the model and find Wannier centers for both topological
# and normal phase of the model
for top_index in ["even","odd"]:
  
  # get the tight-binding model
  my_model=get_kane_mele(top_index)

  # generate list of k-points following some high-symmetry line in
  # the k-space. Variable kpts here is just an array of k-points
  path=[[1.0,0.0],[0.0,1.0]]
  kpts=k_path(path,100)
  
  # initialize figure with subplots
  pl.figure(figsize=(6.0,2.4))
  pl.subplots_adjust(wspace=0.35)
  
  # solve for eigenenergies of hamiltonian on
  # the set of k-points from above
  evals=my_model.solve_all(kpts)
  pl.subplot(1,2,1)
  # plot bands
  pl.plot(evals[0])
  pl.plot(evals[1])
  pl.plot(evals[2])
  pl.plot(evals[3])
  # put title
  pl.title("Kane-Mele: "+top_index+" phase")
  pl.xlabel("k-space")
  pl.ylabel("Energy")
  
  #calculate my-array
  my_array=wf_array(my_model,[41,41])
  
  # solve model on a regular grid, and put origin of
  # Brillouin zone at [-1/2,-1/2]  point
  my_array.solve_on_grid([-0.5,-0.5])
  
  # calculate Berry phases around the BZ in the k_x direction
  # (which can be interpreted as the 1D hybrid Wannier centers
  # in the x direction) and plot results as a function of k_y
  #
  # Following the ideas in
  #   A.A. Soluyanov and D. Vanderbilt, PRB 83, 235401 (2011)
  #   R. Yu, X.L. Qi, A. Bernevig, Z. Fang and X. Dai, PRB 84, 075119 (2011)
  # the connectivity of these curves determines the Z2 index
  #
  wan_cent = my_array.berry_phase([0,1],dir=1,contin=False,berry_evals=True)
  wan_cent/=(2.0*np.pi)
  
  # draw shifted Wannier center positions
  pl.subplot(1,2,2)
  for shift in range(-2,3):
    pl.plot(wan_cent[:,0]+float(shift),"k.")
    pl.plot(wan_cent[:,1]+float(shift),"k.")
  pl.ylim(-1.0,1.0)
  
  pl.ylabel("Wannier center")
  pl.xlabel(r'$k_y$')
  pl.title("1D Wannier centers: "+top_index+" phase")
  
  pl.savefig("kane_mele_"+top_index+".pdf")

print 'Done.\n'

