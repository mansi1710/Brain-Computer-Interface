from keras.models import Sequential
from keras.layers import Dense
import numpy as np 
import scipy.fftpack
from scipy import signal
from sklearn import preprocessing
from scipy.signal import freqz
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

X_raw= np.genfromtxt("dataset_xtrain1.csv", delimiter= ",")
Y= np.genfromtxt("dataset_ytrain.csv", delimiter=",")

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

Y_raw= np.zeros((140,2))
for i in range(140):
    if (Y[i]==1):
        Y_raw[i][0]=1;
        Y_raw[i][1]=0;
    elif (Y[i]==2):
        Y_raw[i][0]=0;
        Y_raw[i][1]=1;

X_raw[0][0]= 0.0103

scaler= preprocessing.StandardScaler().fit(X_raw)
scaler.transform(X_raw)
X_raw1= butter_bandpass_filter(X_raw, 5,20, 128, 5);
np.random.seed(7)

pca = PCA(n_components=60)
pca.fit(X_raw1)
X_raw2= pca.fit_transform(X_raw1)
X_orig=pca.inverse_transform(X_raw2)
print(np.sum((X_raw1- X_orig)**2)/ np.sum(X_raw1**2))
'''
X_train= X_raw2[:126, :]
X_test= X_raw2[126:140, :]
Y_train= Y_raw[:126,:]
Y_test= Y_raw[126:140,:]
'''
plt.figure(1);
plt.plot(X_raw2[0]);
plt.show();

np.random.seed(7)
#define models
model= Sequential()
model.add(Dense(20, input_dim=60, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
#model.add(Dense(25, activation= 'relu'))
#model.add(Dense(10, activation= 'relu'))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_raw2, Y_raw, validation_split=0.2, epochs=2500, batch_size=10)
'''
model.fit(X_train, Y_train, epochs=6, batch_size= 10)
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
'''
