# %%
from utils import string2timestamp
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
# %%
'''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
'''
class MinMaxNormalization(object):
    def __init__(self):
        pass
    def fit(self,x):
        self._min = x.min()
        self._max = x.max()
    def transform(self,x):
        x = 1.*(x-self._min)/(self._max-self._min)
        x = x*2.-1
        return x
    def fit_transform(self,x):
        self.fit(x)
        self.transform(x)
        return x
    def inverse_transform(self,y):
         y = (y+1.)/2
         y = 1. * y * (self._max - self._min) + self._min
         return y

# %%
class STMatrix(object):
    def __init__(self,data,timestamps,T=48,CheckComplete=True):
        super(STMatrix,self).__init__()
        assert len(data)==len(timestamps)
        self.data = data
        self.timestamps = timestamps
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps,self.T)
        if CheckComplete:
            self.check_complete()
        self.make_index()
    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24*60 //self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i-1]+offset!=pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i-1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0
    def make_index(self):
    # mapping timestamp:index
        self.get_index = dict()
        for i,ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i
    def check_it(self,depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True
    def get_matrix(self,timestamps):
        return self.data[self.get_index[timestamps]]
    def create_dataset(self,len_closeness=3,len_period=3,len_trend=3,TrendInterval=7,PeriodInterval=1):
        offset_frame = pd.DateOffset(minutes=24*60 //self.T)
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        '''
        depends = [[1,2,3],48*[1,2,3],48*7*[1,2,3]]
        '''
        depends = [range(1,len_closeness+1),
                   [PeriodInterval*self.T*j for j in range(1,len_period+1)],
                   [TrendInterval*self.T*j for j in range(1,len_trend+1)]]
        i = max(len_closeness,self.T*TrendInterval*len_trend,self.T*PeriodInterval*len_period)
        while i < len(self.pd_timestamps):
            flag = True
            for depend in depends:
                if flag is False:
                    break
                flag = self.check_it([self.pd_timestamps[i]-j*offset_frame for j in depend])
            if flag is False:
                i += 1
                continue
            x_c = [self.get_matrix(self.pd_timestamps[i]-j*offset_frame) for j in depends[0]]
            x_p = [self.get_matrix(self.pd_timestamps[i]-j*offset_frame) for j in depends[1]]
            x_t = [self.get_matrix(self.pd_timestamps[i]-j*offset_frame) for j in depends[2]]
            y = self.get_matrix(self.pd_timestamps[i])
            if len_closeness>0:
                XC.append(np.vstack(x_c))
            if len_period>0:
                XP.append(np.vstack(x_p))
            if len_trend>0:
                XT.append(np.vstack(x_t))
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asanyarray(Y)
        print("XC shape:",XC.shape,"XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC,XP,XT,Y,timestamps_Y

# %%
class STDataSets(Dataset):
    def __init__(self,x,y,device):
        self.xc = torch.from_numpy(np.array(x[0],dtype=np.float32)).to(device)
        self.xp = torch.from_numpy(np.array(x[1],dtype=np.float32)).to(device)
        self.xt = torch.from_numpy(np.array(x[2],dtype=np.float32)).to(device)
        self.ex = torch.from_numpy(np.array(x[3],dtype=np.float32)).to(device)
        self.y = torch.from_numpy(np.array(y,dtype=np.float32)).to(device)
    def __getitem__(self,idx):
        return [self.xc[idx],self.xp[idx],self.xt[idx],self.ex[idx]],self.y[idx]
    def __len__(self):
        return len(self.y)