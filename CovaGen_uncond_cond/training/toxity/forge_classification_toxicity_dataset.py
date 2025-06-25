import pickle
from transvae.rnn_models import RNNAttn
import numpy

with open('/workspace/codes/toxicity/acute_dict_rescale_train.pkl', 'rb') as f:
	train = pickle.load(f)
with open('/workspace/codes/toxicity/acute_dict_rescale_val.pkl', 'rb') as f:
	val = pickle.load(f)
train_value = list(train.values())
ttt = numpy.array(train_value)
cnt = 0
for i in train_value:
	if i <= 2.2:
		cnt+=1
print("done")
