import pandas as pd
from datetime import datetime
import time
import numpy as np
import h5py
import _pickle as pickle
import torch

def string2timestamp(strings,T=48):
    timestamps = []
    hour_per_slot = 24.0/T
    num_per_T = T // 24
    for t in strings:
        year,month,day,slot = int(t[:4]),int(t[4:6]),int(t[6:8]),int(t[8:10])-1
        timestamps.append(pd.Timestamp(datetime(
            year,month,day,hour =int(hour_per_slot*slot),minute=(slot%num_per_T)*int(60.0*hour_per_slot)
        )))
    return timestamps
def timestamp2vec(timestamps):
    vec = [time.strptime(str(t[:8],encoding='utf-8'),'%Y%m%d').tm_wday for t in timestamps]
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i>=5:
            v.append(0)
        else:
            v.append(1)
        ret.append(v)
    return np.asarray(ret)

def saveToh5(X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test):
    f = h5py.File('traintest.h5','w')
    for i in range(4):
        f.create_dataset('X_train_'+str(i),data=X_train[i])
        f.create_dataset('X_test_'+str(i),data=X_test[i])
    f.create_dataset('Y_train',data=Y_train)
    f.create_dataset('Y_test',data=Y_test)
    f.create_dataset('external_dim',data=external_dim)
    f.create_dataset('timestamp_train',data=timestamp_train)
    f.create_dataset('timestamp_test',data=timestamp_test)
    for key in f.keys():
        print(f[key].name)
    f.close()
    
def loadFromh5():
    X_train = []
    X_test = []
    f = h5py.File('traintest.h5','r')
    for i in range(4):
        X_train.append(f['X_train_'+str(i)].value)
        X_test.append(f['X_test_'+str(i)].value)
    Y_test = f['Y_test'].value
    Y_train = f['Y_train'].value
    external_dim = f['external_dim'].value
    timestamp_train = f['timestamp_train'].value
    timestamp_test = f['timestamp_test'].value
    mmn = pickle.load(open('preprocessing.pkl', 'rb'))
    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test
    
def evaluate_model(model,loss,data_iter):
    model.eval()
    l_sum,n = 0.0, 0
    with torch.no_grad():
        for x,y in data_iter:
            y_pred = model(x).view(len(y),-1)
            y = y.view(len(y),-1)
            l = loss(y_pred,y)
            l_sum = l.item()*y.shape[0]
            n += y.shape[0]
        return l_sum / n
    
def evalute_metric(model,data_iter,scaler):
    #print("已修改")
    model.eval()
    with torch.no_grad():
        mae,mape,mse = [],[],[]
        for x,y in data_iter:
             y =scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
             y_pred = scaler.inverse_transform(model(x).view(len(y),-1).cpu().numpy()).reshape(-1)
             assert len(y_pred)==len(y)
             
             d = np.abs(y-y_pred)
             mae += d.tolist()
             #mape += (d/y).tolist() 存在 y = 0的情况
             mape += [ d[i]/y[i] if y[i]!=0 else 0 for i in range(len(y))]
             mse += (d**2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        
        return MAE,MAPE,RMSE
                                  