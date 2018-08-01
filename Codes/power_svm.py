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

X_raw = preprocessing.scale(X_raw)
X_raw1= butter_bandpass_filter(X_raw, 10,40, 128, 5);

X_raw_mu_c3= butter_bandpass_filter(X_raw[:, :1152], 8 ,12,128, 5);
X_raw_beta_c3= butter_bandpass_filter(X_raw[:,:1152], 16, 24 ,128, 5);
X_raw_mu_c4= butter_bandpass_filter(X_raw[:, 1152:2304], 8 ,12,128, 5);
X_raw_beta_c4= butter_bandpass_filter(X_raw[:,1152:2304], 16, 24 ,128, 5);
X_raw_mu_cz= butter_bandpass_filter(X_raw[:, 2304:], 8 ,12,128, 5);
X_raw_beta_cz= butter_bandpass_filter(X_raw[:,2304:], 16 ,24 ,128, 5);

power_mu_c3= np.zeros((140,1 ))
power_mu_c4= np.zeros((140, 1))
power_mu_cz= np.zeros((140, 1))
power_beta_c3= np.zeros((140,1 ))
power_beta_c4= np.zeros((140,1 ))
power_beta_cz= np.zeros((140, 1))
for i in range(140):
    power_mu_c3[i]= np.mean(X_raw_mu_c3[i]**2)
    power_mu_c4[i]= np.mean(X_raw_mu_c4[i]**2)
    power_mu_cz[i]= np.mean(X_raw_mu_cz[i]**2)
    power_beta_c3[i]= np.mean(X_raw_beta_c3[i]**2)
    power_beta_c4[i]= np.mean(X_raw_beta_c4[i]**2)
    power_beta_cz[i]= np.mean(X_raw_beta_cz[i]**2)

Y_raw= np.zeros((140,2))
for i in range(140):
    if (Y[i]==1):
        Y_raw[i][0]=1;
        Y_raw[i][1]=0;
    elif (Y[i]==2):
        Y_raw[i][0]=0;
        Y_raw[i][1]=1;

X_input= np.zeros((140, 6))
for i in range(140):
    X_input[i][0]= power_mu_c3[i];
    X_input[i][1]= power_mu_c4[i];
    X_input[i][2]= power_mu_cz[i];
    X_input[i][3]= power_beta_c3[i];
    X_input[i][4]= power_beta_c4[i];
    X_input[i][5]= power_beta_cz[i];
X_input= preprocessing.scale(X_input)

X_train= X_input[:126, :]
X_test= X_input[126:140, :]
Y_train= Y[:126]
Y_test= Y[126:140]


clf = svm.SVC(C= 1.0, decision_function_shape='ovo', degree= 4, gamma= 4, kernel= 'poly')
clf.fit(X_train, Y_train)
output= clf.predict(X_test)
score= sklearn.metrics.accuracy_score(Y_test, output, normalize= True, sample_weight= None)
print(score*100)
