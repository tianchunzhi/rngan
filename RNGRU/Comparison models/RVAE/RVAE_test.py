import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import os

#定义VAE
class VAE(nn.Module):   #VAE继承自父类：nn.Module
    def __init__(self):
        super(VAE, self).__init__()   #这是对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
        # 定义编码器
        #编码均值
        self.encoder_fc1 = nn.Linear(7,nz) #用于设置网络中的全连接层的，全连接层的输入与输出一般都设置为二维张量，形状通常为[batch_size, size]
        #编码log方差
        self.encoder_fc2 = nn.Linear(7,nz)
        #编码解码器
        self.decoder_fc = nn.Linear(nz+1,7)
        self.Sigmoid = nn.Sigmoid()
    #重构噪声
    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).to(device)   #用来随机生成正态分布数据（128,10）
        z = mean + eps * torch.exp(logvar)
        return z
    #VAE前向传播
    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output
    #定义编码器
    def encoder(self,x):
        x = x
        #调用fc1和fc2两个全连接层，得到均值和方差
        mean = self.encoder_fc1(x)
        logstd = self.encoder_fc2(x)
        z = self.noise_reparameterize(mean, logstd)   #均值和方差结合正态分布噪声得到最终的z
        return z,mean,logstd


    #定义解码器
    def decoder(self,z):
        out3 = self.decoder_fc(z)
        return out3


if __name__ == '__main__':
    nz = 4
    batch_size = 333
    device='cpu'
    print("Random Seed: 88")
    random.seed(88)
    torch.manual_seed(88)
    # 可以优化运行效率
    cudnn.benchmark = True
    print("=====> 构建VAE")
    vae = VAE()
    vae.load_state_dict(torch.load('./CVAE-GAN-VAE.pth', map_location=lambda storage, loc: storage))
    #torch.load('path of your model', map_location=lambda storage, loc: storage)
    # 生成要生成的目标值y
    import copy
    y_target1 = np.random.randint(12, 17, batch_size)  # 生成600-1000内的batch_size个目标值
    y_target = copy.deepcopy(y_target1)
    y_target = y_target.reshape((-1, 1))

    data = np.array(pd.read_csv(r'C:\Users\dell\Desktop/经典数据集/鲍鱼train_人为不平衡.csv', error_bad_lines=False, lineterminator="\n", encoding="gbk"))
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
    outputs = vae.decoder(z)
    # 把属性输出和生成的标签值拼接在一起：

    outputs = outputs.detach().numpy()

    outputs = X_scaler.inverse_transform(outputs)   #将属性反归一化
    y_target1 = y_target1.reshape((-1,1))

    data_all = np.concatenate((outputs, y_target1), 1)


    if not os.path.exists('./img'):
        os.mkdir('./img')
    np.savetxt(r'./img/data_333.csv', data_all,delimiter=',')
