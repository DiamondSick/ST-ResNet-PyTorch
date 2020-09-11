# ST-ResNet-PyTorch
The implementation of ST-ResNet in PyTorch.

**Paper**

[Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction,AAAI 2017](http://export.arxiv.org/pdf/1610.00081)

**Note**

+ The part of loading data is specially written for TaxiBJ. 

+ The model architecture I implemented in st-resnet.py is L2-E-BN.
+ If you want to train the model by yourself, you need to download TaxiBJ in the directory of 'datasets' before. If you want to use the trained model, load the model in the directory of 'save_path'.
+ MAE,MAPE,RMSE on test set are 10.403, 0.256, 17.918.  MAE and RMSE are close to results in the paper, however, MAPE is much larger (MAPE in the paper is about 0.16). I finished this project in hurry and could not figure out what's wrong in my code. If you find the reason, please create an issue and let me know it !!!

**Reference**

[https://github.com/lliony/DeepST-ResNet](https://github.com/lliony/DeepST-ResNet)
