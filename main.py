# %%
from load_data import *
from utils import *
from st_resnet import *
from preprocessing import *
import h5py
import torch.nn as nn
import torch
import os
import logging
# %%
# Hyper Parameters
nb_epoch = 500  
nb_epoch_cont = 100  # number of epoch at training (cont) stage
batch_size = 32  
T = 48  # number of time intervals in one day
lr = 0.0002  
len_closeness = 3  
len_period = 1  
len_trend = 1  
nb_residual_unit = 2
nb_flow = 2
map_height, map_width = 32, 32
days_test = 7 * 4 #last 4 weeks as test 
len_test = T * days_test 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_file = 'log.txt'
lf = logging.FileHandler(log_file,mode='w')
lf.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
lf.setFormatter(formatter)
logger.addHandler(lf)
#logger.info('this is a logger info message')
# %%
#加载数据
path = os.getcwd()
files = os.listdir(path)
if 'traintest.h5' in files:
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = loadFromh5()   
else:
    X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = load_data(
            T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
            preprocess_name='preprocessing.pkl', meta_data=True, meteorol_data=True, holiday_data=True)  
    saveToh5(X_train,Y_train,X_test,Y_test,external_dim,timestamp_train,timestamp_test)
    
# X_train=[XC,XP,XT,meta feature]
# XC:(13728, 6, 32, 32) XP:(13728, 2, 32, 32) XT:(13728, 2, 32, 32) 
# meta feature:(13728, 28)(dayofweek:7,isweekend:1,isholiday:1,windspeed,temperature,weather:16 types)
# Y:(13728, 2, 32, 32)
# %%
train_data = STDataSets(X_train,Y_train,device)
train_iter = torch.utils.data.DataLoader(train_data,batch_size,shuffle=True)
test_data = STDataSets(X_test,Y_test,device)
test_iter = torch.utils.data.DataLoader(test_data,batch_size,shuffle=True)


# %%
c_conf=(len_closeness,nb_flow,map_height,map_width) if len_closeness > 0 else None
p_conf = (len_period, nb_flow, map_height,map_width) if len_period > 0 else None
t_conf = (len_trend, nb_flow, map_height,map_width) if len_trend > 0 else None
loss = nn.MSELoss()
model = ST_ResNet(c_conf=c_conf,p_conf=p_conf,t_conf=t_conf,external_dim=external_dim,nb_residual_unit=nb_residual_unit).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr = lr)

# %%
min_test_loss = np.inf
save_path = 'save_models/model.pt'
for epoch in range(nb_epoch):
    # if epoch >0 :
    #     break
    l_sum,n = 0.0, 0
    model.train()
    for x,y in train_iter:
        # print(x[0].shape,x[1].shape,x[2].shape,x[3].shape,y.shape)
        # break
        y_pred = model(x).view(len(y),-1)
        y = y.view(len(y),-1)
        l = loss(y_pred,y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        l_sum += l.item()*y.shape[0]
        n += y.shape[0]
    test_loss = evaluate_model(model,loss,test_iter)
    if test_loss < min_test_loss:
        min_test_loss = test_loss
        torch.save(model.state_dict(),save_path)
    print("epoch:"+str(epoch)+'\t'+"train_loss:"+str(l_sum / n))
    logger.info("epoch:"+str(epoch)+'\t'+"train_loss:"+str(l_sum / n))
# %% load best model & Evaluation
import pickle
c_conf = (3,2,32,32)
p_conf = (1,2,32,32)
t_conf = (1,2,32,32)
external_dim = 28
nb_residual_unit = 2
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path =  'save_models/model.pt'
loss = nn.MSELoss()

f = h5py.File('traintest.h5')
X_test = [f['X_test_0'].value,f['X_test_1'].value,f['X_test_2'].value,f['X_test_3'].value]
Y_test = f['Y_test'].value
test_data = STDataSets(X_test,Y_test,device)
test_iter = torch.utils.data.DataLoader(test_data,batch_size,shuffle=True)

mmn = pickle.load(open('preprocessing.pkl', 'rb'))

best_model = ST_ResNet(c_conf=c_conf,p_conf=p_conf,t_conf=t_conf,external_dim=external_dim,nb_residual_unit=nb_residual_unit).to(device)
best_model.load_state_dict(torch.load(save_path))

l = evaluate_model(best_model,loss,test_iter)
MAE,MAPE,RMSE= evalute_metric(best_model,test_iter,mmn)

# %%
