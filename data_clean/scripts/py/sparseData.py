from scipy.sparse import csr_matrix
import numpy as np

class DSMat(csr_matrix):
	
	def __init__(self,arg1,shape=None,dtype=None,copy=False):
		if shape is not None:
			csr_matrix.__init__(self,arg1,shape,dtype,copy)
		else:
			csr_matrix.__init__(self,arg1)
		self.eliminate_zeros()

	def norm(self,l='l2'):
		if l=='l2':
			# print((np.sqrt(self.multiply(self)).sum(axis=1)))
			denominator = self.power(2).sum(axis=1)
			denominator[denominator==0]=1
			denominator = np.power(denominator,-0.5)
		
		if l=='l1':
			denominator = self.sum(axis=1)
			denominator[denominator==0]=1
			denominator = np.power(denominator,-1)

		self.__dict__ = DSMat(self.multiply(denominator)).__dict__
		return self


	def write(self,filename,m = 'w'):
		f = open(filename,m)
		(nr,nc) = self.shape
		print(nr, nc, file=f)
		fs = []
		def cols(row):
			(_,col) = row.nonzero()
			s = ' '.join(map(lambda i : "%d:%f"%(i,row[0,i]), col))
			return s
		fs = '\n'.join(map(lambda row: cols(row),self))
		print(fs,file=f)
		f.close()

class SMat:	
	def __new__(self,filename,binary=False,copy=False,dtype=np.float64,dlim=' '):
		if binary==True:
			return self.getBool(filename,dlim=dlim)
		return self.getValue(filename)

	def getValue(filename,copy=False,dtype=np.float64):
		with open(filename,'r') as f:
			nr,nc = map(int,f.readline().strip().split())
			indxptr = []
			indxcol = []
			indxdat = []
			for i,data in enumerate(f):
				_i = str(i)
				for col_data in [x for x in data.strip().split(' ') if x !='']:
					col,dat = col_data.strip().split(':')
					indxcol.append(col)
					indxdat.append(dat)
					indxptr.append(_i)
					del col,dat
				del data,_i
			assert((i+1)==nr)
		return DSMat((np.asarray(indxdat).astype(dtype,copy=False), (np.asarray(indxptr).astype(np.int32,copy=False), np.asarray(indxcol).astype(np.int32,copy=False))),shape=(nr,nc), copy=copy,dtype=dtype)

	def getBool(filename,copy=False,dlim=' '):
		_1 = str(1)
		with open(filename,'r') as f:
			nr,nc = map(int,f.readline().strip().split())
			indxptr = []
			indxcol = []
			indxdat = []
			for i,data in enumerate(f):
				_i = str(i)
				for col_data in [x for x in data.strip().split(dlim) if x !='']:
					col = col_data.strip().split(':')
					indxcol.append(col[0])
					indxdat.append(_1)
					indxptr.append(_i)
					del col
				del data,_i
			assert((i+1)==nr)
		return 


class Data:
	"""docstring for Data"""
	def __init__(self, shape,data):
		super(Data, self).__init__()
		self.shape = shape
		self.data = data
	
	def __getitem__(self,idx):
		return self.data[idx]
				
class LOAD_DATA:
	def __new__(self, file,dtype=np.float64):
		_1 = np.int8(1)
		with open(file,'r') as f:
			nr,ft,lb = map(int,f.readline().strip().split())
			lb_indxptr = []
			lb_indxcol = []
			lb_indxdat = []
			
			ft_indxptr = []
			ft_indxcol = []
			ft_indxdat = []
				
			for i,line in enumerate(f):
				_i = np.int32(i)
				line = [x for x in line.strip().split(' ') if x !='']
	
				if line[0].find(':') !=-1: 
					labels = []
				else:
					labels = list(map(np.int32,line[0].split(',')))
					line = line[1:]
				
				for word in np.unique(line):
					w,v = word.split(':')
					ft_indxcol.append(np.int32(w))
					ft_indxdat.append(dtype(v))
					ft_indxptr.append(_i)
					
				lb_indxcol+=labels
				lb_indxdat+=[_1]*len(labels)
				lb_indxptr+=[_i]*len(labels)
				del line,labels

		return DSMat((ft_indxdat, (ft_indxptr, ft_indxcol)),shape=(nr,ft), copy=False,dtype=dtype),DSMat((lb_indxdat, (lb_indxptr, lb_indxcol)),shape=(nr,lb), copy=False,dtype=np.int8)


class WRITE_DATA:
	"""docstring for SEQ_DATA"""
	def __new__(self, X,Y,file,flag=0):
		indx = np.arange(X.shape[0])
		if flag:
			indx = np.intersect1d(np.unique(X.nonzero()[0]),np.unique(Y.nonzero()[0]))
		with open(file,'w') as f:
			_,ft = X.shape
			_,lb = Y.shape
			nr = indx.shape[0]
			data = []
			print("%d %d %d"%(nr,ft,lb),file=f)
			for k,i in enumerate(indx):
				v = ' '.join(list(map(lambda x:"%d:%f"%(x[0],x[1]),zip(X[i].__dict__['indices'],X[i].__dict__['data']))))
				y = ','.join(list(map(str,Y[i].__dict__['indices'])))
				print('%s %s'%(y,v),file=f)
				print("[%d/%d]"%(k+1,nr),end='\r')

class SEQ_DATA(object):
	"""docstring for SEQ_DATA"""
	def __new__(self, file,norm='l2'):
		documents = []
		_1 = np.int8(1)
		with open(file,'r') as f:
			nr,ft,lb = map(int,f.readline().strip().split())
			indxptr = []
			indxcol = []
			indxdat = []
				
			for i,line in enumerate(f):
				_i = np.int32(i)
				line = [x for x in line.strip().split(' ') if x !='']
	
				if line[0].find(':') !=-1: 
					labels = []
					words = line
	
				else:
					labels = line[0].split(',')
					indxcol+=labels
					indxdat+=[_1]*len(labels)
					indxptr+=[_i]*len(labels)
					line = line[1:]

				if len(line)==0:
					print(i+1)
					exit(0)
				words  = np.zeros((len(line)),dtype=np.int32)
				values = np.zeros((len(line)),dtype=np.float32)

				for i,word in enumerate(line):
					w,v = word.split(':')
					words[i] = np.int32(w)
					values[i] = np.float32(v)
				if norm =='l2':
					values = values/np.sqrt(np.sum(values**2)+1e-16)
				values[values==0]=1
				documents.append([words,values])
				del line
		return Data((nr,ft),documents),DSMat((indxdat, (indxptr, indxcol)),shape=(nr,lb), copy=False,dtype=np.int8)
		