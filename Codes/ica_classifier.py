import numpy as np  
import matplotlib.pyplot as plt 
import csv
from sklearn import preprocessing
import scipy.fftpack
import sklearn
from scipy import signal
from sklearn import svm
from scipy.signal import freqz
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
import mne
from mne.preprocessing import ICA


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

T= 1/128
N= 1152
yf= scipy.fftpack.fft(X_raw[0, :1152])
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.show()


### Mean normalization using sklearn
scaler= preprocessing.StandardScaler().fit(X_raw)
scaler.transform(X_raw)
X_raw1= butter_bandpass_filter(X_raw, 10,40, 128, 5);

T= 1/128
N= 1152
yf= scipy.fftpack.fft(X_raw1 [0, :1152])
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.show()

Y_raw= np.zeros((140,2))
for i in range(140):
	if (Y[i]==1):
		Y_raw[i][0]=1;
		Y_raw[i][1]=0;
	elif (Y[i]==2):
		Y_raw[i][0]=0;
		Y_raw[i][1]=1;

plt.figure(1)
plt.subplot(211)
plt.plot(X_raw[0])
plt.subplot(212)
plt.plot(X_raw1[0])
plt.show()

Y_raw= np.zeros((140,2))
for i in range(140):
	if (Y[i]==1):
		Y_raw[i][0]=1;
		Y_raw[i][1]=0;
	elif (Y[i]==2):
		Y_raw[i][0]=0;
		Y_raw[i][1]=1;

print(X_raw1.shape)

#n_components=X_raw1.shape[1] * self.n_components_rel
ica = FastICA(n_components=3456)
X_raw1=  ica.fit_transform(X_raw1)
print(X_raw1.shape)

plt.figure(4)
plt.subplot(321)
plt.title('ChannelC3- Left movement')
plt.plot(X_raw1.T[0][:1152])
plt.subplot(323)
plt.title('ChannelC4- left movement')
plt.plot(X_raw1.T[0][1152:2304])
plt.subplot(325)
plt.title('ChannelCz- left movement')
plt.plot(X_raw1.T[0][2304:])
plt.subplot(322)
plt.title('ChannelC3- right movement')
plt.plot(X_raw1.T[1][:1152])
plt.subplot(324)
plt.title('ChannelC4- right movement')
plt.plot(X_raw1.T[1][1152:2304])
plt.subplot(326)
plt.title('ChannelCz- right movement')
plt.plot(X_raw1.T[1][2304:])
plt.show()

X_raw2= X_raw1

X_train= X_raw1[:126, :]
X_test= X_raw1[126:140, :]
Y_train= Y[:126]
Y_test= Y[126:140]

clf = svm.SVC(C= 1.0, decision_function_shape='ovo', degree= 4, gamma= 4, kernel= 'poly')
clf.fit(X_train, Y_train)
output= clf.predict(X_test)
score= sklearn.metrics.accuracy_score(Y_test, output, normalize= True, sample_weight= None)
print(score*100)
