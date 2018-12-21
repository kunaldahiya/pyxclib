from xctools.data import data_utils as du
import sys
import os

x,y,num_samples,num_ft,num_lb=du.read_data(sys.argv[1])
output_file = sys.argv[2]
data_set_name = sys.argv[3]
suffix = sys.argv[4]

y = du.binarize_labels(y,num_lb)

directory = os.path.join(output_file,data_set_name)

f=open(directory+'/%s_%s'%(data_set_name,suffix),'w')

for i in range(num_samples):
	indices = y[i].__dict__['indices']
	ft = x[i].__dict__['indices']
	data = x[i].__dict__['data']
	labels = ' '.join(['__label__%d'%yx for yx in indices])
	feature = ' '.join(['%d:%f'%(x,d) for x,d in zip(ft,data)])
	output = '%s  %s'%(labels,feature)
	print(output,file=f)
