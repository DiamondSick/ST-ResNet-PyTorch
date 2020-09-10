# %%
import numpy as np
import h5py
import os
from copy import copy
import _pickle as pickle
from preprocessing import MinMaxNormalization,STMatrix
from utils import timestamp2vec
# %%
# 注意timeslots的字符串是bytes类型的
def load_holiday(timeslots,filepath):
    f = open(filepath,'r',encoding='utf-8')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])
    H = np.zeros(len(timeslots))
    for i,slot in enumerate(timeslots):
        if slot[:8].decode() in holidays:
            H[i] = 1
    print("holidays:",H.sum()/48,'days')
    return H[:,None]
#load_holiday(timestamp_test,'datasets/TaxiBJ/BJ_Holiday.txt')

# %%   
def load_meteorol(timeslots,filepath):
    f = h5py.File(filepath)
    Timeslot = f['date'].value
    Windspeed = f['WindSpeed'].value
    Weather = f['Weather'].value
    Temperature = f['Temperature'].value
    f.close()
    
    M = dict()
    for i,slot in enumerate(Timeslot):
        M[slot] = i
    WS = []
    WR = []
    TE = []
    for slot in timeslots:
        predicted_id = M[slot]
        cur_id = predicted_id-1
        WS.append(Windspeed[cur_id])
        WR.append(Weather[cur_id])
        TE.append(Temperature[cur_id])
    
    WS = np.asarray(WS)
    WR = np.asarray(WR)
    TE = np.asarray(TE)
    
    print("WS.min",WS.min(),"WS.max",WS.max())
    WS = 1.*(WS-WS.min())/ (WS.max() - WS.min())
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())
    
    merge_data = np.hstack([WR,WS[:,None],TE[:,None]])
    return merge_data
        
# %%
'''
data.shape:(4888,2,32,32)
timestamps:[b'2013070101' b'2013070102' b'2013070103' ... b'2013102946' b'2013102947'
 b'2013102948']
'''
def load_stdata(filepath):
    f = h5py.File(filepath,'r')
    data = f['data'].value
    timestamps = f['date'].value
    f.close()
    return data,timestamps
# x,y = load_stdata('datasets/TaxiBJ/BJ13_M32x32_T30_InOut.h5')
# print(y)
# %%
# remove a certain day which has not 48 timestamps
def remove_incomplete_days(data,timestamps,T):
    days = []
    i = 0
    t = []
    while i<len(timestamps):
        t.append(int(timestamps[i][8:]))
        if int(timestamps[i][8:])!=1:
            i+=1
        elif (i+T-1)<len(timestamps) and int(timestamps[i+T-1][8:]) ==T:
            days.append(timestamps[i][:8])
            i += T
        else:
            i += 1
    days = set(days)
    idx = []
    for i,t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)
    data = data[idx]
    timestamps =timestamps[idx]
    # print(data.shape)
    return data,timestamps
#data,timestamps = remove_incomplete_days(x,y,48)       

# %%
def load_data(T=48, nb_flow=2, len_closeness=None, len_period=None, len_trend=None,
              len_test=7*4*48, preprocess_name='preprocessing.pkl',
              meta_data=True, meteorol_data=True, holiday_data=True):
    data_all = []
    timestamps_all = []
    for year in range(13,17):
        filepath = os.path.join('datasets','TaxiBJ','BJ{}_M32x32_T30_InOut.h5'.format(year))
        data,timestamps = load_stdata(filepath)
        data, timestamps = remove_incomplete_days(data, timestamps, T)
        data = data[:,:nb_flow]
        data[data<0] = 0
        data_all.append(data)
        timestamps_all.append(timestamps)
        print(year,data.shape)
    
    data_train = np.vstack(copy(data_all))[:-len_test] #data_train用来归一化
    print('train_data shape:',data_train.shape)
    scaler = MinMaxNormalization()
    scaler.fit(data_train)
    data_all_mmn = [scaler.transform(d) for d in data_all]
    
    #有问题
    fpkl = open(preprocess_name,'wb')
    pickle.dump(scaler,fpkl)
    fpkl.close()
    
    XC,XP,XT = [],[],[]
    Y = []
    timestamps_Y = []
    for data,timestamps in zip(data_all_mmn,timestamps_all):
        st = STMatrix(data,timestamps,T,CheckComplete=False)
        _XC,_XP,_XT,_Y,_timestamps_Y = st.create_dataset(
            len_closeness=len_closeness,len_period=len_period,len_trend=len_trend)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y+=_timestamps_Y
    
    meta_feature = []
    if meta_data:
        time_feature = timestamp2vec(timestamps_Y)
        meta_feature.append(time_feature)
    if holiday_data:
        holiday_feature = load_holiday(timestamps_Y,'datasets/TaxiBJ/BJ_Holiday.txt')
        meta_feature.append(holiday_feature)
    if meteorol_data:
        meteorol_feature = load_meteorol(timestamps_Y,'datasets/TaxiBJ/BJ_Meteorology.h5')
        meta_feature.append(meteorol_feature)
    
    meta_feature = np.hstack(meta_feature) if len(
    meta_feature) > 0 else np.asarray(meta_feature)
    metadata_dim = meta_feature.shape[1] if len(
        meta_feature.shape) > 1 else None
    if metadata_dim < 1:
        metadata_dim = None
    if meta_data and holiday_data and meteorol_data:
        print('time feature:', time_feature.shape, 'holiday feature:', holiday_feature.shape,
              'meteorol feature: ', meteorol_feature.shape, 'mete feature: ', meta_feature.shape)
    
    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape,
          "XT shape: ", XT.shape, "Y shape:", Y.shape)
    
    XC_train, XP_train, XT_train, Y_train = XC[
        :-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[
        -len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[
        :-len_test], timestamps_Y[-len_test:]
    
    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    print('train shape:', XC_train.shape, Y_train.shape,
          'test shape: ', XC_test.shape, Y_test.shape)
    
    if metadata_dim is not None:
        meta_feature_train,meta_feature_test = meta_feature[:-len_test],meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
    
    return X_train,Y_train,X_test,Y_test,scaler,metadata_dim,timestamp_train,timestamp_test
  


# X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = load_data(
#             T=48, nb_flow=2, len_closeness=3, len_period=1, len_trend=1, len_test=7*4*48,
#             preprocess_name='preprocessing.pkl', meta_data=True, meteorol_data=True, holiday_data=True)  

