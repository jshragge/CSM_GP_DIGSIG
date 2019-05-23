#!/usr/bin/env python
import numpy,os

##### User defined data types for Grid IO #####

def dequote(string):
	return(string.strip('"'))

def enquote(string):
	return('"'+string+'"')

# . . Implement a view on a grid
class View:
	ndim=None
	ox=None
	dx=None
	nx=None
	start=None
	stop=None
	step=None
	label=None
	unit=None
	allocated=False

	def __init__(self):
		self.reset()

	def copy(self, other):
		self.ndim=copy.deepcopy(other.ndim)
		self.ox=copy.deepcopy(other.ox)
		self.dx=copy.deepcopy(other.dx)
		self.nx=copy.deepcopy(other.nx)
		self.start=copy.deepcopy(other.start)
		self.stop=copy.deepcopy(other.stop)
		self.step=copy.deepcopy(other.step)
		self.label=copy.deepcopy(other.label)
		self.unit=copy.deepcopy(other.unit)
		self.allocated=copy.deepcopy(other.allocated)

	def allocate(self, ndim):
		self.ndim=ndim
		self.ox=numpy.zeros(ndim,dtype=float)
		self.dx=numpy.ones(ndim,dtype=float)
		self.nx=numpy.ones(ndim,dtype=int)
		self.start=numpy.zeros(ndim, dtype=int)
		self.stop=numpy.ones(ndim,dtype=int)
		self.step=numpy.ones(ndim,dtype=int)
		self.unit=numpy.ndarray(ndim, dtype='object')
		self.unit[:]=""
		self.label=numpy.ndarray(ndim, dtype='object')
		self.label[:]=""
		self.allocated=True

	def reset(self):
		self.ndim=None
		self.ox=None
		self.dx=None
		self.nx=None
		self.start=None
		self.stop=None
		self.step=None
		self.unit=None
		self.label=None
		self.allocated=False

	def fill(self, other, local_dim, other_dim):
		assert local_dim<=self.ndim
		self.ox[local_dim]=other.ox[other_dim]
		self.nx[local_dim]=other.nx[other_dim]
		self.dx[local_dim]=other.dx[other_dim]
		self.start[local_dim]=other.start[other_dim]
		self.stop[local_dim]=other.stop[other_dim]
		self.step[local_dim]=other.step[other_dim]
		self.unit[local_dim]=other.unit[other_dim]
		self.label[local_dim]=other.label[other_dim]

	def create_slices(self):
		# Create slices for use in numpy array operations
		assert(self.allocated)
		slices=[]
		for n in range(0,self.ndim):
			slices.append(slice(self.start[n], self.stop[n], self.step[n]))
		return(tuple(slices))

	def create_slices_from_view(self, ext_view):
		# Create an intersection between this view (self) and an external one (if possible)
		# Doesn't allow for start and stop positions within views
		assert(self.ndim==ext_view.ndim)
		dx_relative_threshold=0.01

		for i in range(0, self.ndim):
			assert(abs((self.dx[i]-ext_view.dx[i])/self.dx[i])<=dx_relative_threshold)

		# start position
		sx_start_intersect=copy.deepcopy(self.ox)
		sx_stop_intersect=copy.deepcopy(self.ox)

		self_start_intersect=self.ox+self.start*self.dx
		self_stop_intersect=self.ox+self.stop*self.dx

		ext_start_intersect=ext_view.ox+ext_view.start*ext_view.dx
		ext_stop_intersect=ext_view.ox+ext_view.stop*ext_view.dx

		nx_intersect=copy.deepcopy(self.nx)

		slices_local=[]
		slices_ext=[]

		badresult=False

		for i in range(0, self.ndim):
			if self.dx[i]>0.0:
				sx_start_intersect[i]=max(self_start_intersect[i],ext_start_intersect[i])
				sx_stop_intersect[i]=min(self_stop_intersect[i], ext_stop_intersect[i])
			else:
				sx_start_intersect[i]=min(self_start_intersect[i], ext_start_intersect[i])
				sx_stop_intersect[i]=max(self_stop_intersect[i], ext_stop_intersect[i])

			nx_intersect[i]=math.floor((sx_stop_intersect[i]-sx_start_intersect[i])/self.dx[i]+0.5)

			if (nx_intersect[i]<=0):
				badresult=True
			else:
				slices_local.append(slice(int(math.floor((sx_start_intersect[i]-self.ox[i])/self.dx[i]+0.5)),int(math.floor((sx_stop_intersect[i]-self.ox[i])/self.dx[i]+0.5)),self.step[i]))
				slices_ext.append(slice(int(math.floor((sx_start_intersect[i]-ext_view.ox[i])/ext_view.dx[i]+0.5)),int(math.floor((sx_stop_intersect[i]-ext_view.ox[i])/ext_view.dx[i]+0.5)),ext_view.step[i]))

		if badresult:
			return([None, None])
		else:
			return([slices_local, slices_ext])

	def create_view_from_slices(self, slices):
		# Create a view on this array given some slices
		assert(self.allocated)
		assert(len(slices)==len(self.nx))
		view=View()
		view.allocate(self.ndim)

		view.dx=copy.deepcopy(self.dx)
		view.label=copy.deepcopy(self.label)
		view.unit=copy.deepcopy(self.unit)

		for i in range(0, self.ndim):
			view.nx[i]=abs(slices[i].stop-slices[i].start)
			view.ox[i]=self.ox[i]+slices[i].start*self.dx[i]
			view.start[i]=0
			view.stop[i]=view.nx[i]
			view.step[i]=slices[i].step

		return(view)


	def default_view(self, dim):
		# Make up a default view for dimension dim
		assert(self.allocated)
		self.start[dim]=0
		self.stop[dim]=self.nx[dim]
		self.step[dim]=1

	def make_default_view(self):
		for n in range(0, self.ndim):
			self.default_view(n)

	def create_dict(self):
		# Create a dictionary for json upload etc.
		assert(allocated);
		parm=dict()
		parm["ndim"]=copy.deepcopy(ndim)
		parm["nx"]=copy.deepcopy(self.nx)
		parm["ox"]=copy.deepcopy(self.ox)
		parm["dx"]=copy.deepcopy(self.dx)
		parm["start"]=copy.deepcopy(self.start)
		parm["stop"]=copy.deepcopy(self.stop)
		parm["step"]=copy.deepcopy(self.step)
		parm["unit"]=copy.deepcopy(self.unit)
		parm["label"]=copy.deepcopy(self.label)
		return(parm)

	def unload_dict(self, parm):
		# Download from dictionary
		self.ndim=copy.deepcopy(parm["ndim"])
		self.nx=copy.deepcopy(parm["nx"])
		self.ox=copy.deepcopy(parm["ox"])
		self.dx=copy.deepcopy(parm["dx"])
		self.start=copy.deepcopy(parm["start"])
		self.stop=copy.deepcopy(parm["stop"])
		self.step=copy.deepcopy(parm["step"])
		self.unit=copy.deepcopy(parm["unit"])
		self.label=copy.deepcopy(parm["label"])

	def print_metadata(self, stream):
		print >> stream, "ndim=", self.ndim
		print >> stream, "ox=", self.ox
		print >> stream, "dx=", self.dx
		print >> stream, "nx=", self.nx
		print >> stream, "start=", self.start
		print >> stream, "stop=", self.stop
		print >> stream, "step=", self.step
		print >> stream, "unit=", self.unit
		print >> stream, "label=", self.label
		
# .. Define Grid Class
class Grid():
	# The numpy array holding the data
	view=View()

	# an array or view to a binary file
	data=None

	# are we allocated?
	allocated=False
	dtype=numpy.float32

	# Array ordering
	order="C"

	def deallocate(self):
		self.data=none
		self.dtype=numpy.float32
		self.view.deallocate()
		gc.collect()
		self.allocated=False

	def allocate(self, dtype=numpy.float32, order="C"):
		assert(self.view.allocated)
		self.order=order
		self.dtype=dtype
		self.data=numpy.zeros(self.view.nx, dtype=self.dtype, order=self.order)
		self.allocated=True

	def reset(self):
		self.view = View()
		self.data=[]
		self.allocated=False
		self.dtype=numpy.float32

	def __init__(self):
		self.reset()

	def ingest_array(self, array, binary_order="C"):
		self.view.ndim=int(len(array.shape))
		self.view.allocate(len(array.shape))
		self.view.nx[:]=numpy.array(array.shape, dtype=int)
		self.view.ox[:]=numpy.zeros((self.view.ndim), dtype=numpy.float32)
		self.view.dx[:]=self.view.ox[:]+1.0
		if binary_order!=self.order:
			self.data=array.ravel(order=self.order).reshape(self.view.nx, order=self.order)
		else:
			self.data=array

		self.allocated=True

	def ingest_binary(self, binary_fname, dtype=numpy.float32, binary_order="C"):
		# If view is already there, ingest a binary file with the proper checks
		assert(self.view.allocated)
		self.dtype=dtype
		if binary_order!=self.order:
			temp=numpy.fromfile(binary_fname, dtype=self.dtype).reshape(tuple(self.view.nx), order=binary_order)
			self.data=temp.ravel(order=self.order).reshape(tuple(self.view.nx), order=self.order)
		else:
			self.data=numpy.fromfile(binary_fname, dtype=self.dtype).reshape(self.view.nx, order=self.order)
			
# Read rsf file
def read_rsf_file(infile=None, use_memmap=False):
	if (infile==None):
		# File is from standard input
		tempgrid=read_rsf(sys.stdin, use_memmap)
	else:
		# Try opening the file
		input_file=str(infile).strip()
		if os.path.isfile(input_file):
			# Open the file
			f=open(input_file,'r')
			tempgrid=read_rsf(f, use_memmap)
			f.close()
		else:
			# It might be a tag
			f=open(input_file+".rsf", 'r')
			tempgrid=read_rsf(f, use_memmap)
			f.close()

	return(tempgrid)
	
def read_rsf(instream, use_memmap=False):
    # Read rsf file from a stream filelike object
    parm=dict()

    for line in instream:
        part="".join(line.split()).partition('=')
        if (part[2]!=''):
            parm[part[0]]=dequote(part[2])
            
    # Get the number of dimensions
    count=1
    ndim=0
    while "n"+str(count) in parm.keys():
        ndim+=1
        count+=1
    
    # Get the data size
    if "esize" in parm:
        esize=int(parm["esize"])
    else:
        esize=4

    # Get the data type
    if "type" in parm:
        type=parm["type"]
    else:
        type=None

    if "data_format" in parm:
        data_format=parm["data_format"]
    else:
        data_format="native_float"

    # get the form
    if "form" in parm:
        form=parm["form"]
    else:
        form="native"

    # Get the right dtype
    if ((type=="int" or data_format=="native_int") and esize==4):
        dtype=numpy.int32
    elif ((type=="complex" or data_format=="native_complex") and esize==8):
        dtype=numpy.complex64
    elif ((type=="short" or data_format=="native_short") and esize==2):
        dtype=numpy.int16
    else:
        dtype=numpy.float32

    # Get the input grid
    ingrid=Grid()
    ingrid.view.allocate(ndim)

    for i in range(0,ndim):
        var='o'+str(i+1)
        if var in parm:
            ingrid.view.ox[i]=float(parm[var])
        var='d'+str(i+1)
        if var in parm:
            ingrid.view.dx[i]=float(parm[var])
        var='n'+str(i+1)
        if var in parm:
            ingrid.view.nx[i]=int(parm[var])
        var='label'+str(i+1)
        if var in parm:
            ingrid.view.label[i]=parm[var]
        var='unit'+str(i+1)
        if var in parm:
            ingrid.view.unit[i]=parm[var]
    
    # Make sure we have an input binary file
    assert("in" in parm)

    # Strip the quotes
    parm["in"]=parm["in"].strip('"')

    # Now read the data
    if use_memmap:
        # Use the efficient memory map format
        ingrid.data=numpy.memmap(parm["in"], dtype=dtype, mode='r', order='F', shape=tuple(ingrid.view.nx))
    elif (form=="native"):
        ingrid.data=numpy.fromfile(parm["in"], dtype=dtype).reshape(ingrid.view.nx, order='F')
    else:
        # Try reading from ascii
        ingrid.data=numpy.fromfile(parm["in"], dtype=dtype, sep=" ").reshape(ingrid.view.nx, order='F')

    # This has allocated Grid
    ingrid.allocated=True
    ingrid.dtype=dtype

    return(ingrid)
    
    			