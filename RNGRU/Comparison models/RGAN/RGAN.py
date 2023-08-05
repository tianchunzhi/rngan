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


#定义生成器
class generator(nn.Module):   #VAE继承自父类：nn.Module
    def __init__(self):
        super(generator, self).__init__()   #这是对继承自父类的属性进行初始化。而且是用父类的初始化方法来初始化继承的属性。
        self.decoder_fc = nn.Sequential(nn.Linear(nz+2,6), nn.ReLU(),nn.Linear(6,8))   #判别器是分类器，需要加sigmoid激活函数
    #生成器前向传播
    def forward(self, x):
        output = self.decoder_fc(x)
        return output

class Discriminator(nn.Module):
    def __init__(self,outputn=1):
        super(Discriminator, self).__init__()
        self.fd = nn.Sequential(
            nn.Linear(8, outputn),   #判别器的输入只有数据，没有标签
            nn.Sigmoid()   #判别器是分类器，需要加sigmoid激活函数
        )

    def forward(self, input):
        x = input
        x = self.fd(x)
        return x


class Classifier(nn.Module):
    def __init__(self,outputn=1):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(8, 5),   #判别器的输入只有数据，没有标签
            nn.Linear(5, outputn),
            nn.Sigmoid()   #分类器，需要加sigmoid激活函数
        )

    def forward(self, input):
        x = input
        x = self.fc(x)
        return x

if __name__ == '__main__':
    batchSize = 200
    nz = 4
    nepoch=800
    if not os.path.exists('./img_CVAE-GAN'):
        os.mkdir('./img_CVAE-GAN')
    print("Random Seed: 88")
    random.seed(88)
    torch.manual_seed(88)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 可以优化运行效率
    cudnn.benchmark = True

    data = np.array(pd.read_csv(r'D:\研二寒假\课题2\不平衡分类数据集\yeast4/train_分层.csv', error_bad_lines=False, lineterminator="\n", encoding="gbk"))

    #对原始数据进行标准化：
    datax = data[:,:-1]
    datay = data[:,-1]
    # 打乱顺序：
    index = [i for i in range(len(datax))]
    np.random.shuffle(index)
    datax = datax[index, :]
    datay = datay[index].reshape((-1,1))

    # 特征缩放：归一化
    from sklearn.preprocessing import StandardScaler
    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()
    datax_norm = X_scaler.fit_transform(datax)
    datax = torch.from_numpy(datax_norm)
    datay_norm = Y_scaler.fit_transform(datay)
    datay = torch.from_numpy(datay_norm)
    dataset = TensorDataset(datax,datay)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batchSize,shuffle=True)
    #用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor，后续只需要再包装成Variable即可作为模型的输入
    print("=====> 构建VAE")
    G = generator().to(device)
    print("=====> 构建D")
    D = Discriminator(1).to(device)
    print("=====> 构建C")
    C = Classifier(1).to(device)   #这个是一个标签分类器
    criterion = nn.BCELoss().to(device)   #对一个batch里面的数据做二元交叉熵并且求平均
    MSECriterion = nn.MSELoss().to(device)   #对一个banch里面的数据求均方误差损失并求平均

    print("=====> Setup optimizer")
    optimizerD = optim.Adam(D.parameters(), lr=0.0005)
    optimizerC = optim.Adam(C.parameters(), lr=0.0005)
    optimizerG = optim.Adam(G.parameters(), lr=0.0005)
    err_d = []
    err_c = []
    err_g = []
    for epoch in range(nepoch):
        #定义三个用来存放D,C,G误差的列表：

        for i, (data,label) in enumerate(dataloader, 0):   #data：（200，10）因为banchsize = 200(dataloader),开始索引为0
            # 属性
            data = data.type(torch.float32).to(device)
            # 真实样本类别
            label = label.type(torch.float32).to(device)   #改变数据类型，label不用再进行编码了，因为分类器是回归的
            batch_size = data.shape[0]
            # 定义真实样本类别的独热编码：
            #label = label.type(torch.long)  # 改变数据类型
            #label_onehot_r = torch.zeros((batch_size, 2)).to(device)  # (128,2)标签种类为2
            #label_onehot_r[torch.arange(batch_size).type(torch.long), label] = 1  # 相当于对标签进行独热编码

            #生成样本目标值
            y_target = np.ones(batch_size)

            #y_target = y_target.reshape((-1, 1))
            #y_target = Y_scaler.transform(y_target)
            y_target = torch.from_numpy(y_target)
            y_target = y_target.type(torch.float32).to(device)
            y_target_one = y_target.type(torch.long)  # 改变数据类型

            # 定义生成样本类别的独热编码：
            label_onehot_f = torch.zeros((batch_size, 2)).to(device)  # (128,2)标签种类为2


            label_onehot_f[torch.arange(batch_size).type(torch.long), y_target_one] = 1  # 相当于对标签进行独热编码

            # 先训练C
            output = C(data)  #data：（128,10）
            #将分类器的结果转化为0或1：
            for e in range(batch_size):
                if output[e] >= 0.5:
                    output[e]=1
                elif output[e] < 0.5:
                    output[e]=0

            errC = MSECriterion(output, label)   #均方误差
            C.zero_grad()   #初始化梯度
            errC.backward()   #梯度反向传播
            optimizerC.step()   #参数更新
            # 再训练D
            output = D(data)
            output = output.reshape(-1)   #转化为一列
            #将判别器的输出转化为0或1;
            for f in range(batch_size):
                if output[f] >= 0.5:
                    output[f]=1
                elif output[f] < 0.5:
                    output[f]=0

            real_label = torch.ones(batch_size).to(device)   # 定义真实的图片label为1
            fake_label = torch.zeros(batch_size).to(device)  # 定义假的图片的label为0
            errD_real = criterion(output, real_label)   #得到判别真实样本的误差


            ###########################这里可以改：把解码器的输入改为噪声（五维）+标签编码（两维）
            z = torch.randn(batch_size, nz).to(device)
            z = torch.cat([z,label_onehot_f],1)
            fake_data = G.forward(z)   #输出的是（128,10）
            output = D(fake_data)   #（128,1）
            output = output.reshape(-1)
            for m in range(batch_size):
                if output[m] >= 0.5:
                    output[m]=1
                elif output[m] < 0.5:
                    output[m]=0

            errD_fake = criterion(output, fake_label)   #得到判别生成样本的误差

            errD = errD_real+errD_fake    #得到判别真实样本和生成样本的误差和
            D.zero_grad()   #初始化判别器D的梯度值
            errD.backward()
            optimizerD.step()   #更新判别器D的参数值

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_C: %.4f '% (epoch, nepoch, i, len(dataloader),errD.item(),errC.item()))

        err_d.append(errD.item())
        err_c.append(errC.item())

        #取出单元素张量的元素值并返回该值,精度比较好


# import matplotlib.pyplot as plt
# x = range(nepoch)
# y1 = err_d
# y2 = err_c
#
# plt.plot(x,y1,color='r')
# plt.plot(x,y2,color='black')
#
# plt.legend()
# plt.show()


torch.save(G.state_dict(), './CVAE-GAN-VAE_train.pth')
# torch.save(D.state_dict(),'./CVAE-GAN-Discriminator.pth')
# torch.save(C.state_dict(),'./CVAE-GAN-Classifier.pth')