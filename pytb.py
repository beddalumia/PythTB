# PyTB python tight binding module
# Version 1.5, May 25, 2012

# Copyright 2010, 2012 by Sinisa Coh and David Vanderbilt
# 
# This file is part of PyTB.  PyTB is free software: you can
# redistribute it and/or modify it under the terms of the GNU General
# Public License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# 
# PyTB is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
# 
# A copy of the GNU General Public License should be available
# alongside this source in a file named gpl-3.0.txt.  If not,
# see <http://www.gnu.org/licenses/>.
# 
# PyTB is availabe at http://www.physics.rutgers.edu/~dhv/pytb/

import numpy as nu # numerics for matrices
import sys # for exiting
import copy # for deepcopying
try:
  import pylab as pl # for plotting
except:
  print "Unable to load pylab! tbmodel.visualize(...) will not work. "

class tbmodel:
  """Class for general kind of tight-binding hamiltonian"""

  def __init__(self,dimk,dimr,lat,orb,per=None):
    """Initialize variables and read in geometry"""

    # initialize _dimk = dimensionality of k-space (integer)
    if type(dimk).__name__!='int':
      raise Exception("Argument dimk not an integer")
    if dimk < 0 or dimk > 4:
      raise Exception("Argument dimk out of range. Must be between 0 and 4.")
    self._dimk=dimk

    # initialize _dimr = dimensionality of r-space (integer)
    if type(dimr).__name__!='int':
      raise Exception("Argument dimr not an integer")
    if dimr < dimk or dimr > 4:
      raise Exception("Argument dimr out of range. Must be dimr>=dimk and dimr<=4.")
    self._dimr=dimr

    # initialize _lat = lattice vectors, array of dimr*dimr
    #   format is _lat(lat_vec_index,cartesian_index)
    # special option: 'unit' implies unit matrix
    if lat=='unit':
        self._lat=nu.identity(dimr,float)
    elif type(lat).__name__ not in ['list','ndarray']:
      raise Exception("Argument lat is not a list.")
    else:
      self._lat=nu.array(lat,dtype=float)
      if self._lat.shape!=(dimr,dimr):
        raise Exception("Wrong lat array dimensions")

    # initialize _norb = number of basis orbitals per cell
    #   and       _orb = orbital locations, in reduced coordinates
    #   format is _orb(orb_index,lat_vec_index)
    # special option: 'bravais' implies one atom at origin
    if (orb=='bravais'):
      self._norb=1
      self._orb=nu.zeros((1,dimr))
    elif type(orb).__name__ not in ['list','ndarray']:
      raise Exception("Argument orb is not a list")
    else:
      self._orb=nu.array(orb,dtype=float)
      if len(self._orb.shape)!=2:
        raise Exception("Wrong orb array rank")
      self._norb=self._orb.shape[0] # number of orbitals
      if self._orb.shape[1]!=dimr:
        raise Exception("Wrong orb array dimensions")

    # choose which self._dimk out of self._dimr dimensions are
    # to be considered periodic.
    if per==None:
      # by default first _dimk dimensions are periodic
      self._per=range(self._dimk)
    else:
      if len(per)!=self._dimk:
        raise Exception("Wrong choice of periodic/infinite direction!")
      # store which directions are the periodic ones
      self._per=per

    # Initialize onsite energies to zero
    self._site_energies=nu.zeros(self._norb,dtype=float)

    # Initialize hoppings to empty list
    self._hoppings=[]

    # The onsite energies and hoppings are not specified
    # when creating a 'tbmodel' object.  They are speficied
    # subsequently by separate function calls defined below.

  def set_sites(self,en_list):
    """function to define site energies"""
    if (len(en_list)!=self._norb):
      raise Exception("Wrong number of site energies")
    self._site_energies=nu.array(en_list)

  def add_hop(self,amp,i,j,lvec=None):
    """function to define hopping terms in tb hamiltonian"""
    # for each pair of orbitals, input the hopping in one direction only
    # (the other will automatically be added)
    #   amp = hopping amplitude (float or complex)
    #   i = index of orbital in home cell (int)
    #   j = index of orbital in neighboring cell (int)
    #   lvec (list of length dimk) = indices of lattice vector
    #     pointing to neighboring cell
    #     if dimk=0, lvec doesn't need to be specified, if specified it is ignored
    #     if dimk=1, lvec can be a simple integer
    # This specifies the following matrix element <i|H|j+R>
    #    which is strictly speaking, hopping from site j+R to site i
    #
    if self._dimk!=0 and lvec==None:
      raise Exception("Need to specify lvec!")
    #
    # make sure that if <i|H|j+R> is specified that <j|H|i-R> is not!
    for h in self._hoppings:
      if i==h[2] and j==h[1]:
        if self._dimk==0:
          raise Exception(\
"""Following matrix element was already implicitely specified:  i="""+str(i)+" j="+str(j)+"""
  Remember, specifying <i|H|j> automatically specifies <j|H|i>.
  You need to specify only one of the two!""")
        elif False not in (nu.array(lvec)==(-1)*nu.array(h[3])):
          raise Exception(\
"""Following matrix element was already implicitely specified: i="""+str(i)+" j="+str(j)+" R="+str(lvec)+"""
  Remember, specifying <i|H|j+R> automatically specifies <j|H|i-R>.
  You need to specify only one of the two!""")
    # store new hopping
    if self._dimk==0:
      hop=[amp,int(i),int(j)]
    # if necessary convert from integer to list of length one
    elif self._dimk==1 and type(lvec).__name__=='int':
      hop=[amp,int(i),int(j),[lvec]]
    else:
      hop=[amp,int(i),int(j),lvec]
      # currently no type or dimension checking on lvec; fix?
    self._hoppings.append(hop)

  def display(self):
    """function to display tight-binding model"""
    print '---------------------------------------'
    print 'report of tight-binding model'
    print '---------------------------------------'
    print 'k-space dimension   =',self._dimk
    print 'r-space dimension   =',self._dimr
    print 'periodic directions =',self._per
    print 'number of orbitals  =',self._norb
    print 'lattice vectors:'
    for i,o in enumerate(self._lat):
      print " #",nice_int(i,2)," ===>  [",
      for j,v in enumerate(o):
        print nice_float(v,7,4),
        if j!=len(o)-1:
          print ",",
      print "]"
    print 'positions of orbitals:'
    for i,o in enumerate(self._orb):
      print " #",nice_int(i,2)," ===>  [",
      for j,v in enumerate(o):
        print nice_float(v,7,4),
        if j!=len(o)-1:
          print ",",
      print "]"
    print 'site energies:'
    for i,site in enumerate(self._site_energies):
      print " #",nice_int(i,2)," ===>  ",
      print nice_float(site,7,4)
    print 'hoppings:'
    for i,hopping in enumerate(self._hoppings):
      print "<",nice_int(hopping[1],2),"| H |",nice_int(hopping[2],2),
      if len(hopping)==4:
        print "+ [",
        for j,v in enumerate(hopping[3]):
          print nice_int(v,2),
          if j!=len(hopping[3])-1:
            print ",",
          else:
            print "]",
      print ">     ===> ",
      print nice_float(complex(hopping[0]).real,7,4),
      if complex(hopping[0]).imag<0.0:
        print " - ",
      else:
        print " + ",      
      print nice_float(abs(complex(hopping[0]).imag),7,4),
      print " i"
    print
    
  def visualize(self,dir1,dir2=None,eig_dr=None,draw_hoppings=True):
    """Draws a graph of tight-binding model.
    Figure is projected along directions dir1 and dir2
    no matter how many dimensions actual model has.
    Optionally, also draws some eigenstate."""
    # start a new figure
    fig=pl.figure(figsize=(6,6))
    
    # make both axis the same
    pl.axis("equal")
    
    def proj(v):
      "Project vector onto drawing plane"
      coord_x=v[dir1]
      if dir2==None:
        coord_y=0.0
      else:
        coord_y=v[dir2]
      return [coord_x,coord_y]

    def to_cart(red):
      "Convert reduced to Cartesian coordinates"
      return nu.dot(red,self._lat)

    # draw origin
    pl.plot([0.0],[0.0],"bo",mec="w",zorder=7)

    # first draw unit cell vectors which are considered to be periodic
    for i in self._per:
      # pick a unit cell vector and project it down to the drawing plane
      vec=proj(self._lat[i])
      pl.plot([0.0,vec[0]],[0.0,vec[1]],"b-",lw=1.5,zorder=7)

    # now draw all orbitals
    for i in range(self._norb):
      # find position of orbital in cartesian coordinates
      pos=to_cart(self._orb[i])
      pos=proj(pos)
      pl.plot([pos[0]],[pos[1]],"ro",mec="w",zorder=10)

    # draw hopping terms
    if draw_hoppings==True:
      for h in self._hoppings:
        # draw both i->j+R and i-R->j hop
        for s in range(2):
          # get "from" and "to" coordinates
          pos_i=self._orb[h[1]]
          pos_j=self._orb[h[2]]
          # add also lattice vector if not 0-dim
          if self._dimk!=0:
            if s==0:
              pos_j=pos_j+nu.array(h[3])
            if s==1:
              pos_i=pos_i-nu.array(h[3])
          # project down vector to the plane
          pos_i=nu.array(proj(to_cart(pos_i)))
          pos_j=nu.array(proj(to_cart(pos_j)))
          # add also one point in the middle to bend the curve
          prcnt=0.05 # bend always by this ammount
          pos_mid=(pos_i+pos_j)*0.5
          dif=pos_j-pos_i # difference vector
          orth=nu.array([dif[1],-1.0*dif[0]]) # orthogonal to difference vector
          orth=orth/nu.sqrt(nu.dot(orth,orth)) # normalize
          pos_mid=pos_mid+orth*prcnt*nu.sqrt(nu.dot(dif,dif)) # shift mid point in orthogonal direction
          # draw hopping
          all_pnts=nu.array([pos_i,pos_mid,pos_j]).T
          pl.plot(all_pnts[0],all_pnts[1],"g-",lw=0.5,zorder=8)
          # draw "from" and "to" sites
          pl.plot([pos_i[0]],[pos_i[1]],"o",c=[0.8,0.6,0.6],mec="w",zorder=9)
          pl.plot([pos_j[0]],[pos_j[1]],"o",c=[0.8,0.6,0.6],mec="w",zorder=9)

    # now draw the eigenstate
    if eig_dr!=None:
      for i in range(self._norb):
        # find position of orbital in cartesian coordinates
        pos=to_cart(self._orb[i])
        pos=proj(pos)
        # find norm of eigenfunction at this point
        nrm=(eig_dr[i]*eig_dr[i].conjugate()).real
        # rescale and get size of circle
        nrm_rad=2.0*nrm*float(self._norb)
        pl.plot([pos[0]],[pos[1]],"bo",mec="w",ms=nrm_rad,zorder=11,alpha=0.7)

    # center the image
    #  first get the current limit, which is probably tight
    xl=pl.xlim()
    yl=pl.ylim()
    # now get the center of current limit
    centx=(xl[1]+xl[0])*0.5
    centy=(yl[1]+yl[0])*0.5
    # now get the maximal size (lengthwise or heightwise)
    mx=max([xl[1]-xl[0],yl[1]-yl[0]])
    # set new limits
    extr=0.05 # add some boundary as well
    pl.xlim(centx-mx*(0.5+extr),centx+mx*(0.5+extr))
    pl.ylim(centy-mx*(0.5+extr),centy+mx*(0.5+extr))
    
    # return a figure to the user
    return fig
    
  def _genHam(self,kInput=None):
    """Generate hamiltonian for a certain k-point,
    K-point is given in reduced coordinates!"""
    kpnt=nu.array(kInput)
    if kInput!=None:
      # if kpnt is just a number then convert it to an array    
      if len(kpnt.shape)==0:
        kpnt=nu.array([kpnt])    
      # check that k-vector is of corect size
      if kpnt.shape!=(self._dimk,):
        raise Exception("k-vector of wrong shape!")
    else:
      if self._dimk!=0:
        raise Exception("Have to provide a k-vector!")        
    # zero the hamiltonian matrix
    ham=nu.zeros((self._norb,self._norb),dtype=complex)
    # modify diagonal elements
    for i in range(self._norb):
      ham[i,i]=self._site_energies[i]
    # go over all hoppings
    for hopping in self._hoppings:
      # get all data for the hopping parameter
      amp=complex(hopping[0])
      i=hopping[1]
      j=hopping[2]
      # in 0-dim case there is no phase factor
      if self._dimk>0:
        lvec=nu.array(hopping[3],dtype=float)
        # vector from one site to another
        rv=-self._orb[i,:]+self._orb[j,:]+lvec
        # Take only components of vector which are periodic
        rv=rv[self._per]
        # Calculate the hopping, see details in info/tb/tb.pdf
        phase=nu.exp((2.0j)*nu.pi*nu.dot(kpnt,rv))
        amp=amp*phase
      # add this hopping into a matrix and also its conjugate
      ham[i,j]+=amp
      ham[j,i]+=amp.conjugate()
    return ham

  def _solHam(self,ham,eig_vectors=False):
    """Solves hamiltonian and returns eigenvectors, eigenvalues"""
    #solve matrix
    if eig_vectors==False: # only find eigenvalues
      eval=nu.linalg.eigvalsh(ham)
      # sort eigenvalues and convert to real numbers
      eval=nicefy_eig(eval)        
      return nu.array(eval,dtype=float)
    else: # find eigenvalues and eigenvectors
      (eval,eig)=nu.linalg.eigh(ham)
      # transpose matrix eig since otherwise it is confusing
      # now eig[i,:] is eigenvector for eval[i]-th eigenvalue
      eig=eig.T
      # sort evectors, eigenvalues and convert to real numbers
      (eval,eig)=nicefy_eig(eval,eig)        
      return (eval,eig)

  def solve_all(self,klist=None,eig_vectors=False):
    """Solves for band energies on a list of kpoints.
    eig_vectors tells you whether to return eigenvectors as well.

    SC: Since we have reg_array now, this function is maybe obsolete!

    """
    # if not 0-dim case
    if klist!=None:
      nkp=len(klist) # number of k points
      # first initialize matrices for all return data
      #    indices are [band,kpoint]
      ret_eval=nu.zeros((self._norb,nkp),dtype=float)
      #    indices are [band,kpoint,orbital]
      ret_evec=nu.zeros((self._norb,nkp,self._norb),dtype=complex)
      # go over all kpoints
      for i,k in enumerate(klist):
        # generate hamiltonian at that point
        ham=self._genHam(k)
        # solve hamiltonian
        if eig_vectors==False:
          eval=self._solHam(ham,eig_vectors=eig_vectors)
          ret_eval[:,i]=eval[:]
        else:
          (eval,evec)=self._solHam(ham,eig_vectors=eig_vectors)
          ret_eval[:,i]=eval[:]
          ret_evec[:,i,:]=evec[:,:]
      # return stuff
      if eig_vectors==False:
        # indices of eval are [band,kpoint]
        return ret_eval
      else:
        # indices of eval are [band,kpoint] for evec are [band,kpoint,orbital]
        return (ret_eval,ret_evec)
    else: # 0 dim case
      # generate hamiltonian
      ham=self._genHam()
      # solve
      if eig_vectors==False:
        eval=self._solHam(ham,eig_vectors=eig_vectors)
        # indices of eval are [band]
        return eval
      else:
        (eval,evec)=self._solHam(ham,eig_vectors=eig_vectors)
        # indices of eval are [band] and of evec are [band,orbital]
        return (eval,evec)

  def solve_one(self,kpoint=None,eig_vectors=False):
    """Solves for band energies at one kpoint.
    eig_vectors tells you whether to return eigenvectors as well."""
    # if not 0-dim case
    if kpoint!=None:
      if eig_vectors==False:
        eval=self.solve_all([kpoint],eig_vectors=eig_vectors)
        # indices of eval are [band]
        return eval[:,0]
      else:
        (eval,evec)=self.solve_all([kpoint],eig_vectors=eig_vectors)
        # indices of eval are [band] for evec are [band,orbital]
        return (eval[:,0],evec[:,0,:])
    else:
      # do the same as solve_all
      return self.solve_all(eig_vectors=eig_vectors)

  def cut_piece(self,num,fin_dir,glue_edgs=False):
    """Function to cut out a finite piece of model out of infinite model. It
    returns a new object of type tbmodel corresponding to the cutout.
    Direction which becomes finite is defined with fin_dir and number of unitcells
    with num.
    Variable glue_edgs defines whether to add hoppings from one finite edge to another or not.
    Orbitals in cutout model are numbered so that i-th orbital of the n-th unit
    cell is i+norb*n.
    """
    if self._dimk ==0:
      raise Exception("Model is already finite")
    if type(num).__name__!='int':
      raise Exception("Argument num not an integer")

    # generate orbitals of a finite model
    fin_orb=[]
    onsite=[] # store also onsite energies
    for i in range(num): # go over all cells in finite direction
      for j in range(self._norb): # go over all orbitals in one cell
        # make a copy of j-th orbital
        orb_tmp=nu.copy(self._orb[j,:])
        # change coordinate along finite direction
        orb_tmp[fin_dir]+=float(i)
        # add to the list
        fin_orb.append(orb_tmp)
        # do the onsite energies at the same time
        onsite.append(self._site_energies[j])
    fin_orb=nu.array(fin_orb)

    # generate periodic directions of a finite model
    fin_per=copy.deepcopy(self._per)
    # find if list of periodic directions contains the one you
    # want to make finite
    if fin_per.count(fin_dir)!=1:
      raise Exception("Can not make model finite along this direction!")
    # remove index which is no longer periodic
    fin_per.remove(fin_dir)    

    # generate object of tbmodel type that will correspond to a cutout
    fin_model=tbmodel(self._dimk-1,
                      self._dimr,
                      copy.deepcopy(self._lat),
                      fin_orb,
                      fin_per)

    # now put all onsite terms for the finite model
    fin_model.set_sites(onsite)
       
    # put all hopping terms
    for c in range(num): # go over all cells in finite direction
      for h in range(len(self._hoppings)): # go over all hoppings in one cell
        # amplitude of the hop is the same
        amp=self._hoppings[h][0]

        # lattice vector of the hopping
        lvec=copy.deepcopy(self._hoppings[h][3])
        jump_fin=lvec[fin_dir] # store by how many cells is the hopping in finite direction
        if fin_model._dimk!=0:
          lvec[fin_dir]=0 # one of the directions now becomes finite
          
        # index of "from" and "to" hopping indices
        hi=self._hoppings[h][1] + c*self._norb
        #   have to compensate  for the fact that lvec in finite direction
        #   will not be used in the finite model
        hj=self._hoppings[h][2] + (c + jump_fin)*self._norb 

        # decide whether this hopping should be added or not
        add_hop=True
        #  if jumping below bottom edge
        if c==0 and jump_fin<0: 
          if glue_edgs==True:
            # if edges are glued, do the hop and correct the index
            add_hop=True
            hj+=self._norb*num
          else:
            add_hop=False
        #  if jumping above top edge
        elif c==num-1 and jump_fin>0: 
          if glue_edgs==True:
            # if edges are glued, do the hop and correct the index
            add_hop=True
            hj-=self._norb*num
          else:
            add_hop=False
            
        # add hopping to a finite model
        if add_hop==True:
          if fin_model._dimk==0:
            fin_model.add_hop(amp,hi,hj)
          else:
            fin_model.add_hop(amp,hi,hj,lvec)

    return fin_model

  def reduceDim(self,kdim,kval):
    """Reduce dimensionality of the model in k-space. This is
    different from cutting the model in real-space. This function will
    return copy of this object where direction kdim will not be
    periodic and hoppings will be as if you had k-vector along that
    component equal to kval."""
    #
    if self._dimk==0:
      raise Exception("Can not reduce dimensionality even further!")
    # make a copy
    ret=copy.deepcopy(self)
    # make one of the directions not periodic
    ret._per.remove(kdim)
    ret._dimk=len(ret._per)
    # check that really removed one and only one direction
    if ret._dimk!=self._dimk-1:
      raise Exception("Specified wrong dimension to reduce!")
    # modify all hopping parameters for this value of kval
    for h in range(len(self._hoppings)):
      hop=self._hoppings[h]
      amp=complex(hop[0])
      i=hop[1]; j=hop[2]
      lvec=nu.array(hop[3],dtype=float)
      # vector from one site to another
      rv=-ret._orb[i,:]+ret._orb[j,:]+lvec
      # take only r-vector component along direction you are not making periodic
      rv=rv[kdim]
      # Calculate the part of hopping phase, only for this direction
      phase=nu.exp((2.0j)*nu.pi*(kval*rv))
      ret._hoppings[h][0]=amp*phase
    return ret
  
class wf_array:
  """The class to store wave functions on an array
  and calculate the Berry phase, 1st Chern number, etc."""
  def __init__(self,model,meshar):
    """Initialize object with a given model and dimension of array.
    Form of parameter meshar should be [nx] for 1-D or [nx,ny] for 2-D"""
    # store orbitals from the model
    self._norb=model._norb
    self._orb=nu.copy(model._orb)
    # store entire model as well
    self._model=copy.deepcopy(model)
    # store dimension of array of points on which to keep wavefunctions
    self._meshar=nu.array(meshar)
    self._dimar=len(self._meshar)
    # generate temporary array used later to generate object ._wfs
    wfs_dim=nu.copy(self._meshar)
    wfs_dim=nu.append(wfs_dim,self._norb)
    wfs_dim=nu.append(wfs_dim,self._norb)
    # store wavefunctions here in the form _wfs[kx_index,ky_index,band,orb]
    self._wfs=nu.zeros(wfs_dim,dtype=complex)

  def solve_on_grid(self,start_k):
    """Solve model on a mesh of k-points starting from point start_k.
    Also imposes PBC because this thing is always solved on entire BZ anyways.
    """
    if self._dimar==1:
      for i in range(self._meshar[0]):
        # generate a kpoint
        kpt=[start_k[0]+float(i)/float(self._meshar[0]-1)]
        # solve at that point
        (eval,evec)=self._model.solve_one(kpt,eig_vectors=True)
        # store wavefunctions
        self.add_wf([i],evec)
      # impose boundary conditions
      self.impose_pbc([True])
    elif self._dimar==2:
      for i in range(self._meshar[0]):
        for j in range(self._meshar[1]):
          kpt=[start_k[0]+float(i)/float(self._meshar[0]-1),\
               start_k[1]+float(j)/float(self._meshar[1]-1)]
          (eval,evec)=self._model.solve_one(kpt,eig_vectors=True)
          self.add_wf([i,j],evec)
      self.impose_pbc([True,True])
    elif self._dimar==3:
      for i in range(self._meshar[0]):
        for j in range(self._meshar[1]):
          for k in range(self._meshar[2]):
            kpt=[start_k[0]+float(i)/float(self._meshar[0]-1),\
                 start_k[1]+float(j)/float(self._meshar[1]-1),\
                 start_k[2]+float(k)/float(self._meshar[2]-1)]
            (eval,evec)=self._model.solve_one(kpt,eig_vectors=True)
            self.add_wf([i,j,k],evec)
      self.impose_pbc([True,True,True])
    else:
      raise Exception("Wrong dimensionality!")

  def add_wf(self,loc,vec):
    """Adds one wave function to the proper location in wf_array._wfs
    User needs to specify the location [i,j...] as parameter loc"""
    #if len(loc)!=self._dimar:
    #  raise Exception("Wrong dimension for location")
    #for i,n in enumerate(self._meshar):
    #  if loc[i]<-1 or loc[i]>n-1:
    #    raise Exception("Location out of range")
    #if vec.shape!=(self._norb,self._norb):
    #  raise Exception("Wrong dimension of vector")
    
    # Store vector in array
    # TODO: this part can be written in a nicer way
    if len(loc)==1:
      self._wfs[loc[0]]=nu.copy(vec)
    elif len(loc)==2:
      self._wfs[loc[0],loc[1]]=nu.copy(vec)
    elif len(loc)==3:
      self._wfs[loc[0],loc[1],loc[2]]=nu.copy(vec)
    else:
      raise Exception("Wrong dimensionality of location!")

  def impose_pbc(self,bc):    
    """This function impose the periodic boundary condition to
    wf_array.  Variable bc has the form ['True','False',..] where
    False means do nothing and True at i-th position means to apply
    periodic boundary condition in i-th direction assuming that the
    k-point mesh jumps by exactly +1.0 * G_i. This is not checked because
    wf_array does not store k-vectors at which model was solved. This maybe
    should be fixed in future versions"""    
    #if len(bc)!=self._dimar:
    #  raise Exception("Wrong dimension for Boundary Condition")
      
    # TODO: this part can be written in a nicer way
    #
    # Impose periodic boundary condition on wavefunctions
    #= 1-D case
    if self._dimar==1:
      if bc[0]==True:
        for i in range(self._norb):
          self._wfs[-1,:,i]=self._wfs[0,:,i]*nu.exp(-2.j*nu.pi*self._orb[i][0])
    #= 2-D case
    elif self._dimar==2:
      if bc[0]==True:
        for i in range(self._norb):
          self._wfs[-1,:,:,i]=self._wfs[0,:,:,i]*nu.exp(-2.j*nu.pi*self._orb[i][0])
      if bc[1]==True:        
        for i in range(self._norb):
          self._wfs[:,-1,:,i]=self._wfs[:,0,:,i]*nu.exp(-2.j*nu.pi*self._orb[i][1])
    #= 3-D case
    elif self._dimar==3:
      if bc[0]==True:
        for i in range(self._norb):
          self._wfs[-1,:,:,:,i]=self._wfs[0,:,:,:,i]*nu.exp(-2.j*nu.pi*self._orb[i][0])
      if bc[1]==True:        
        for i in range(self._norb):          
          self._wfs[:,-1,:,:,i]=self._wfs[:,0,:,:,i]*nu.exp(-2.j*nu.pi*self._orb[i][1])
      if bc[2]==True:
        for i in range(self._norb):
          self._wfs[:,:,-1,:,i]=self._wfs[:,:,0,:,i]*nu.exp(-2.j*nu.pi*self._orb[i][2])
    else:
      raise Exception("Wrong dimensionality!")
      
  def berry_phase(self,occ,dir=None,contin=False):
    """The function to calculate Berry phase, returns number between -pi and pi.
    The dir specifies to calculate Berry phase along which direction.
    The occ specifies which bands are occupied.
    The contin specifies whether berry phase should be continuous or not."""
    #if dir<0 or dir>self._dimar-1:
    #  raise Exception("Direction key out of range")

    # 1D case
    if self._dimar==1:
      # pick which wavefunctions to use
      wf_use=self._wfs[:,occ,:]
      # calculate berry phase
      pha=calc_one_phase(wf_use)
    # 2D case
    elif self._dimar==2:
      # choice along which direction you wish to calculate berry phase
      if dir==0:
        pha=[]
        for i in range(self._meshar[1]):
          wf_use=self._wfs[:,i,:,:][:,occ,:]
          pha.append(calc_one_phase(wf_use))
      elif dir==1:
        pha=[]
        for i in range(self._meshar[0]):
          wf_use=self._wfs[i,:,:,:][:,occ,:]
          pha.append(calc_one_phase(wf_use))
      else:
        raise Exception("Wrong direction for Berry phase calculation!")
    # 3D case
    elif self._dimar==3:
      # choice along which direction you wish to calculate berry phase
      if dir==0:
        pha=[]
        for i in range(self._meshar[1]):
          pha_t=[]
          for j in range(self._meshar[2]):
            wf_use=self._wfs[:,i,j,:,:][:,occ,:]
            pha_t.append(calc_one_phase(wf_use))
          pha.append(pha_t)
      elif dir==1:
        pha=[]
        for i in range(self._meshar[0]):
          pha_t=[]
          for j in range(self._meshar[2]):
            wf_use=self._wfs[i,:,j,:,:][:,occ,:]
            pha_t.append(calc_one_phase(wf_use))
          pha.append(pha_t)
      elif dir==2:
        pha=[]
        for i in range(self._meshar[0]):
          pha_t=[]
          for j in range(self._meshar[1]):
            wf_use=self._wfs[i,j,:,:,:][:,occ,:]
            pha_t.append(calc_one_phase(wf_use))
          pha.append(pha_t)
      else:
        raise Exception("Wrong direction for Berry phase calculation!")
    else:
      raise Exception("Wrong dimensionality!")

    # convert phases to numpy array
    if self._dimar>1:
      pha=nu.array(pha,dtype=float)

    # iron out 2pi jumps, make the gauge choice such that first phase in the
    # list is fixed, others are then made continuous.
    if contin==True:
      # 2D case
      if self._dimar==2:
        pha=iron_out(pha,pha[0])
      # 3D case
      elif self._dimar==3:
        for i in range(pha.shape[1]):
          if i==0: clos=pha[0,0]
          else: clos=pha[0,i-1]
          pha[:,i]=iron_out(pha[:,i],clos)
      elif self._dimar!=1:
        raise Exception("Wrong dimensionality!")

    return pha

  def berry_curv(self,occ):
    """Calculate Berry curvature for a 2dim mesh of points by calculating
    Berry phase around the boundary of this 2dim mesh."""
    # 2D case
    if self._dimar==2:
      curv=0.0
      # sum over all small squares
      for i in range(self._meshar[0]-1):
        for j in range(self._meshar[1]-1):
          # generate a small loop made out of four pieces
          wf_use=nu.zeros((5,len(occ),self._wfs.shape[-1]),dtype=complex)
          wf_use[0]=self._wfs[i,j,:,:][occ,:]
          wf_use[1]=self._wfs[i+1,j,:,:][occ,:]
          wf_use[2]=self._wfs[i+1,j+1,:,:][occ,:]
          wf_use[3]=self._wfs[i,j+1,:,:][occ,:]
          wf_use[4]=self._wfs[i,j,:,:][occ,:]
          # calculate phase around one square
          curv+=calc_one_phase(wf_use)
    else:
      raise Exception("Wrong dimensionality!")

    return curv

def k_path(kpts,nk):
  """Makes path in k-space along which the Hamiltonian will be solved"""
  if kpts=='full':
    # this means the full Brillouin zone for 1D case
    return nu.array(range(nk+1),dtype=float)/nk
  elif kpts=='half':
    # this means the half Brillouin zone for 1D case
    return nu.array(range(nk+1),dtype=float)/(2*nk)
  else:
    # now we know we are in 2D or 3D
    ret=[]
    k_list=nu.array(kpts)
    # go over all kpoints
    for i in range(len(k_list)-1):
      # go over all steps
      for j in range(nk):
        cur=k_list[i]+(k_list[i+1]-k_list[i])*float(j)/float(nk)
        ret.append(cur)
    # add last point
    ret.append(k_list[-1])
    return nu.array(ret)

def nicefy_eig(eval,eig=None):
  "Sort eigenvaules and eigenvectors, if given, and convert to real numbers"
  # first take only real parts of the eigenvalues
  eval=nu.array(eval.real,dtype=float)
  # sort energies
  args=eval.argsort()
  eval=eval[args]
  if eig!=None:
    eig=eig[args]
    return (eval,eig)
  return eval
    
# for nice justified printout
def nice_float(x,just,rnd):
  return str(round(x,rnd)).rjust(just)
def nice_int(x,just):
  return str(x).rjust(just)

def wf_dpr(wf1,wf2):
  """calculate dot product between two wavefunctions.
  wf1 and wf2 are of the form [orbital]"""
  return nu.dot(wf1.conjugate(),wf2)

def calc_one_phase(wf):
  """Do one Berry phase calculation (also returns a product of M matrices).
  Always return number between -pi and pi.
  wf has format [kpnt,band,orbital] and kpnt has to be one dimensional.
  Assumes that first and last k-point are the same. Therefore if
  there are n wavefunctions in total, will calculate phase along n-1 links only!"""
  # number of occupied states
  nocc=wf.shape[1]
  # Two temporary matrices
  prd=nu.identity(nocc,dtype=complex)
  ovr=nu.zeros([nocc,nocc],dtype=complex)
  # go over all pairs of k-points, assuming that last point is overcounted!
  for i in range(wf.shape[0]-1):
    # generate overlap matrix, go over all bands
    for j in range(nocc):
      for k in range(nocc):
        ovr[j,k]=wf_dpr(wf[i,j,:],wf[i+1,k,:])
    # multiply overlap matrices
    prd=nu.dot(prd,ovr)
  # calculate Berry phase
  det=nu.linalg.det(prd)
  pha=nu.angle(det)    
  return pha

def iron_out(arr,clos):
  """Reads in 1d array of numbers _arr_ and makes sure that they are
  continuous, i.e.  that there are no jumps of 2pi. First number is
  made as close to _clos_ as possible."""
  # go through entire list and "iron out" 2pi jumps
  for i in range(len(arr)):
    # which number to compare to
    if i==0: cmpr=clos
    else: cmpr=arr[i-1]
    # check if there is a 2pi jump
    while abs(cmpr-arr[i])>nu.pi:
      if cmpr-arr[i]>nu.pi:
        arr[i]+=2.0*nu.pi
      elif cmpr-arr[i]<-1.0*nu.pi:
        arr[i]-=2.0*nu.pi
  return arr
