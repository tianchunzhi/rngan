import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import os

class generator(nn.Module):   
    def __init__(self):
        super(generator, self).__init__()   
        self.decoder_fc = nn.Sequential(nn.Linear(nz+1,6), nn.ReLU(),nn.Linear(6,7))   
    def forward(self, x):
        output = self.decoder_fc(x)
        return output

if __name__ == '__main__':
    nz = 4
    batch_size = 200
    device='cpu'
    print("Random Seed: 88")
    random.seed(88)
    torch.manual_seed(88)
    cudnn.benchmark = True
    print("=====> 构建VAE")
    G = generator()
    G.load_state_dict(torch.load('./CVAE-GAN-VAE_train.pth'))
    import copy
    y_target1 = np.random.randint(15, 29, batch_size)  # 生成600-1000内的batch_size个目标值
    y_target = copy.deepcopy(y_target1)
    y_target = y_target.reshape((-1, 1))

    data = np.array(pd.read_csv(r'C:\Users\dell\Desktop\鲍鱼数据集/鲍鱼train_分层.csv', error_bad_lines=False, lineterminator="\n", encoding="gbk"))
    datax = data[:, :-1]
    datay = data[:,-1]

    from sklearn.preprocessing import StandardScaler

    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()
    datax_norm = X_scaler.fit_transform(datax)
    datay = datay.reshape((-1, 1))
    datay_norm = Y_scaler.fit_transform(datay)
    #这个地方应该用原始数据的标准化还是生成值的标准化？？？？？？
    y_target = Y_scaler.transform(y_target)
    y_target = torch.from_numpy(y_target)
    y_target = y_target.type(torch.float32).to(device)


    z = torch.randn((batch_size, 4))
    z = torch.cat([z,y_target],1)
    outputs = G.forward(z)
    # 把属性输出和生成的标签值拼接在一起：

    outputs = outputs.detach().numpy()

    outputs = X_scaler.inverse_transform(outputs)   #将属性反归一化
    y_target1 = y_target1.reshape((-1,1))

    data_all = np.concatenate((outputs, y_target1), 1)


    if not os.path.exists('./img'):
        os.mkdir('./img')
    np.savetxt(r'C:\Users\dell\Desktop\200.csv', data_all,delimiter=',')
