#在训练生成器时的输入：噪声+目标值
#训练判别器的输入：属性
#训练分类器的输入：属性
#所以原始样本的目标值不用独热编码

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
import heapq
import warnings
warnings.filterwarnings("ignore")


#定义生成器
class generator(nn.Module):   #VAE继承自父类：nn.Module
    def __init__(self):
        super(generator, self).__init__()
        #generator1:RNGRU
        # RNGRU-GRU

        self.RN_fc = nn.Linear(2 * 7, 1)
        # RNGRU-GRU
        self.GRU_gru = torch.nn.GRU(input_size=7, hidden_size=7, num_layers=3)
        self.Sigmoid = nn.Sigmoid()

        #generator2
        self.generator2 = nn.Sequential(nn.Linear(nz+1,6), nn.ReLU(),nn.Linear(6,7))
        # 判别器是分类器，需要加sigmoid激活函数

        #λ1和λ2的取值：
        self.λ1 = nn.Sequential(nn.Linear(7,1), nn.Sigmoid())

    #生成器前向传播
    def forward(self, N, R, nz):
        x = torch.cat((N, R), dim=1)
        sig = self.Sigmoid(self.RN_fc(x))
        o_1 = sig * N + (1 - sig) * R
        o_2, _ = self.GRU_gru(o_1)
        x2 = self.generator2(nz)
        x = self.λ1(x2)
        x = x * o_2 + (1 - x) * x2
        return x

class Discriminator(nn.Module):
    def __init__(self,outputn=1):
        super(Discriminator, self).__init__()
        self.fd = nn.Sequential(
            nn.Linear(7, outputn),   #判别器的输入只有数据，没有标签
            nn.Sigmoid()   #判别器是分类器，需要加sigmoid激活函数
        )

    def forward(self, input):
        x = input
        x = self.fd(x)
        return x


class Predictor(nn.Module):
    def __init__(self,outputn=1):
        super(Predictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(7, 5),   #判别器的输入只有数据，没有标签
            nn.Linear(5, outputn),
            nn.Sigmoid()   #
        )

    def forward(self, input):
        x = input
        x = self.fc(x)
        return x


if __name__ == '__main__':
    batchSize = 200
    nepoch=100
    nz = 4
    if not os.path.exists('./img_CVAE-GAN'):
        os.mkdir('./img_CVAE-GAN')
    print("Random Seed: 88")
    random.seed(88)
    torch.manual_seed(88)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 可以优化运行效率
    cudnn.benchmark = True
    data = np.array(pd.read_csv(r'D:\Pycharmwenjian\RNGRU\CVAE\CVAE\鲍鱼train_分层.csv', error_bad_lines=False, lineterminator="\n", encoding="gbk"))
    #对原始数据进行标准化：
    datax = data[:,:-1]
    datay = data[:,-1]
    datay_f = np.array(datay, copy=True)
    datax_f = np.array(datax, copy=True)


    def finddata(mbz,l):
        huigui = []
        jinlin = []
        for j in datay_f:
            n_1 = []
            n = abs(j - mbz)
            n_1.append(n)
        min_number_hg = heapq.nsmallest(l, n_1)
        min_index_hg = []
        for t in min_number_hg:
            index_hg = n_1.index(t)
            min_index_hg.append(index_hg)
        huigui = datax_f[min_index_hg]
        zuijinlin_index1 = n_1.index(min(n_1))
        zuijinlin_sx = datax_f[zuijinlin_index1]
        for i in range(datax_f.shape[0]):
            m_1 = []
            m = abs(datax_f[i,:] - zuijinlin_sx)
            m_1.append(m)
        min_number_jl = heapq.nsmallest(l, m_1)
        min_index_jl = []
        for t in min_number_jl:
            index_jl = m_1.index(t)
            min_index_jl.append(index_jl)
        jinlin = datax_f[min_index_jl]
        return np.array(huigui),np.array(jinlin)
    # 打乱顺序：
    index = [i for i in range(len(datax))]
    np.random.shuffle(index)
    datax = datax[index, :]
    datay = datay[index].reshape((-1,1))

    # 特征缩放：归一化

    from sklearn.preprocessing import MinMaxScaler
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()
    datax = torch.from_numpy(X_scaler.fit_transform(datax))
    datay = torch.from_numpy(Y_scaler.fit_transform(datay))
    dataset = TensorDataset(datax,datay)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batchSize,shuffle=True)
    #用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor，后续只需要再包装成Variable即可作为模型的输入
    print("=====> 构建generator")
    generator = generator().to(device)
    print("=====> 构建D")
    dis = Discriminator().to(device)
    print("=====> 构建P")
    pre = Predictor().to(device)
    MSECriterion = nn.MSELoss().to(device)   #对一个banch里面的数据求均方误差损失并求平均
    print("=====> Setup optimizer")
    optimizergenerator = optim.Adam(generator.parameters(), lr=0.0005)
    optimizerD = optim.Adam(dis.parameters(), lr=0.0005)
    optimizerP = optim.Adam(pre.parameters(), lr=0.0005)

    for epoch in range(nepoch):
        #定义三个用来存放D,C,G误差的列表：
        for i, (data,label) in enumerate(dataloader, 0):   #data：（200，10）因为banchsize = 200(dataloader),开始索引为0
            # 先处理一下数据
            data = data.type(torch.float32).to(device)
            label = label.type(torch.float32).to(device)   #改变数据类型，label不用再进行编码了，因为分类器是回归的
            for i in range(2000):
                pre_out = pre(data)
                err1 = MSECriterion(pre_out, label)  # 均方误差
                pre.zero_grad()  # 初始化梯度
                err1.backward(retain_graph=True)  # 梯度反向传播
                optimizerP.step()  # 参数更新
            batch_size = data.shape[0]
            # 训练GRU
            mubiaozhi_sc = torch.randint(2, 29, (1,))
            mubiaozhi_sc = torch.reshape(mubiaozhi_sc, (1,1))
            z = torch.randn(1, nz)
            z = torch.cat([z,mubiaozhi_sc],dim=1)
            huigui,jinlin = finddata(mubiaozhi_sc, 12)
            output = generator.forward(torch.from_numpy(huigui).to(torch.float32),torch.from_numpy(jinlin).to(torch.float32),z)  #data：（128,10）

            # 再训练D
            output11 = dis(output)
            output11 = output11.reshape(-1)  # 转化为一列
            real_label = torch.ones(batch_size).to(device).reshape(-1)  # 定义真实的图片label为1
            fake_label = torch.zeros(batch_size).to(device).reshape(-1)  # 定义假的图片的label为0
            errD_fake = MSECriterion(output11, fake_label)  # 得到判别真实样本的误差

            output12 = dis(data)  # （128,1）
            output12 = output12.reshape(-1)
            errD_real = MSECriterion(output12, real_label)  # 得到判别生成样本的误差

            errD = errD_real + errD_fake  # 得到判别真实样本和生成样本的误差和
            dis.zero_grad()  # 初始化判别器D的梯度值
            errD.backward(retain_graph=True)
            optimizerD.step()  # 更新判别器D的参数值

            #训练generator

            output21 = dis(output)
            output21 = output21.reshape(-1)  # 转化为一列
            real_label = torch.ones(batch_size).to(device).reshape(-1)  # 定义真实的图片label为1
            fake_label = torch.zeros(batch_size).to(device).reshape(-1)  # 定义假的图片的label为0
            errG_fake = MSECriterion(output21, real_label)  # 得到判别真实样本的误差

            output22 = dis(data)  # （128,1）
            output22 = output12.reshape(-1)
            errG_real = MSECriterion(output22, fake_label)  # 得到判别生成样本的误差
            mubiaozhi_sc = torch.randint(2, 29, (batch_size,))
            mubiaozhi_sc = torch.reshape(mubiaozhi_sc, (batch_size, 1))
            mubiaozhi_bzh = torch.from_numpy(Y_scaler.fit_transform(mubiaozhi_sc))
            shuxing_bzh = torch.from_numpy(Y_scaler.fit_transform(mubiaozhi_sc))

            z = torch.randn(batch_size, nz)
            z = torch.cat([z, mubiaozhi_sc], dim=1)
            huigui1, jinlin1 = finddata(mubiaozhi_sc, 12)
            outputz = generator.forward(torch.from_numpy(huigui1).to(torch.float32),torch.from_numpy(jinlin1).to(torch.float32), z)  # data：（128,10）
            shuxing_bzh = torch.from_numpy(X_scaler.fit_transform(outputz.detach().numpy()))
            yuce = pre(shuxing_bzh)
            err_P = MSECriterion(yuce.to(torch.float32), mubiaozhi_bzh.to(torch.float32))
            err_G = errG_fake+errG_real+err_P
            generator.zero_grad()  # 初始化所有的梯度值
            err_G.backward()  # 反向传播
            optimizergenerator.step()  # 梯度更新

        print('[%d/%d] Loss: %.4f'% (epoch, nepoch,err_G.item()))
    yangben = torch.empty(1, 8)
    for k in range(200):
        yangben1 = torch.empty(1,8)
        mubiaozhi_cs = torch.randint(25, 30, (1,))
        mubiaozhi_cs = torch.reshape(mubiaozhi_cs, (1, 1))
        z = torch.randn(1, nz)
        z = torch.cat([z, mubiaozhi_cs], dim=1)
        huigui,jinlin = finddata(mubiaozhi_cs, 12)
        output = generator.forward(torch.from_numpy(huigui).to(torch.float32),
                                    torch.from_numpy(jinlin).to(torch.float32), z)  # data：（128,10）
        yangben1 = torch.cat((output, mubiaozhi_cs), dim=1)
        yangben = torch.cat((yangben, yangben1), dim=0)
    yangben = yangben.detach().numpy()
    np.savetxt(r'D:\Pycharmwenjian\RNGRU\RNGAN\shengchengyangben\200.csv', yangben)