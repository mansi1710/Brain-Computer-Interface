import numpy as np
import csv
from sklearn import preprocessing
from scipy.signal import freqz
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

X_raw= np.genfromtxt("dataset_xtrain1.csv", delimiter= ",")
Y= np.genfromtxt("dataset_ytrain.csv", delimiter=",")
X_raw[0][0]= 0.0103

X_raw = preprocessing.scale(X_raw)
X_raw= butter_bandpass_filter(X_raw, 10, 40, 128, 5);

Y_raw= np.zeros((140,2))
for i in range(140):
	if (Y[i]==1):
		Y_raw[i][0]=1;
		Y_raw[i][1]=0;
	elif (Y[i]==2):
		Y_raw[i][0]=0;
		Y_raw[i][1]=1;

X_train= X_raw[:126, :]
X_test= X_raw[126:140, :]
Y_train= Y_raw[:126,:]
Y_test= Y_raw[126:140,:]

input_layerSize= X_train.shape[1]
hidden_layerSize= 5
output_layerSize= Y_train.shape[1]
epoch = 1200
lr= 0.001


def sigmoid(x):
	return 1/(1+np.exp(-x))
	
def derivative_sigmoid(x):
	return x*(1-x)
	
def cost(y, t):
	return np.sum(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))

np.random.seed(42)
wh=  np.random.normal(0, scale= 0.1, size= (input_layerSize, hidden_layerSize))
wout= np.random.normal(scale= 0.1, size= (hidden_layerSize, output_layerSize))
bh= np.random.normal(scale=0.1, size= (1, hidden_layerSize))
bout= np.random.normal(scale= 0.1, size= (1, output_layerSize))

#print(wh)
#print(X.shape)
#print(wh.shape)

acc_test= np.zeros((epoch,));
acc_train= np.zeros((epoch,));

def accuracy(x, y):
	hidden_layer_input1= np.dot(x, wh)
	hidden_layer_input= hidden_layer_input1 + bh	
	hidden_layer_activation = sigmoid(hidden_layer_input)
	output_layer_input1 = np.dot(hidden_layer_activation, wout)
	output_layer_input = output_layer_input1+  bout
	output= sigmoid(output_layer_input)	
	c=0
	for i in range(x.shape[0]):
		for j in range(2):
			if output[i][j]>0.5:
				output[i][j]=1
			else:
				output[i][j]=0
			if output[i][j]==Y_train[i][j]:
					c+=1
	return (c/(x.shape[0]*2))	


def training(x, y, wout, wh, bout, bh, x_test, y_test):
	hidden_layer_input1= np.dot(x, wh)
	hidden_layer_input= hidden_layer_input1 + bh
	hidden_layer_activation = sigmoid(hidden_layer_input)
	output_layer_input1 = np.dot(hidden_layer_activation, wout)
	output_layer_input = output_layer_input1+  bout
	output= sigmoid(output_layer_input)

	#back propogation
	E= y- output
	error= np.sum(E**2)/126
	d_output= E* derivative_sigmoid(output)
	wout+= hidden_layer_activation.T.dot(d_output)*lr
	error_hidden_layer= d_output.dot(wout.T)
	d_hidden_layer= error_hidden_layer* derivative_sigmoid(hidden_layer_activation)
	wh+= X_train.T.dot(d_hidden_layer)*lr
	bout= bout+ np.sum(d_output, axis= 0)*lr
	bh+= np.sum(d_hidden_layer, axis=0)* lr
	test_acc= accuracy(x_test, y_test);
	train_acc= accuracy(x, y);
	return wout, wh, bout, bh, train_acc, test_acc

for i in range(epoch):
	wout, wh, bout, bh, acc_train[i], acc_test[i]= training(X_train, Y_train, wout, wh, bout, bh, X_test, Y_test);
	#forward feedback
	
trainacc= accuracy(X_train, Y_train);
testacc= accuracy(X_test, Y_test);
print(trainacc);
print(testacc);

plt.figure(1)
plt.subplot(211)
plt.plot(acc_train);
plt.title('Training accuracy')

plt.subplot(212)
plt.plot(acc_test);
plt.title('Testing accuracy')
plt.show()