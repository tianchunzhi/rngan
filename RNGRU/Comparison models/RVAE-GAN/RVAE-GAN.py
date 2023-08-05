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

class Classifier(nn.Module):
    def __init__(self,outputn=1):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(7, 5),   #判别器的输入只有数据，没有标签
            nn.Linear(5, outputn)
        )

    def forward(self, input):
        x = input
        x = self.fc(x)
        return x

def loss_function(recon_x,x,mean,logstd):
    MSE = MSECriterion(recon_x,x)   #均方根损失
    # 因为var是标准差的自然对数，先求自然对数然后平方转换成方差
    var = torch.pow(torch.exp(logstd),2)   #方差
    KLD = -0.5 * torch.sum(1+torch.log(var)-torch.pow(mean,2)-var)   #KL散度
    return MSE+KLD

if __name__ == '__main__':
    batchSize = 100
    nz = 4
    nepoch=550
    if not os.path.exists('./img_CVAE-GAN'):
        os.mkdir('./img_CVAE-GAN')
    print("Random Seed: 88")
    random.seed(88)
    torch.manual_seed(88)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 可以优化运行效率
    cudnn.benchmark = True

    data = np.array(pd.read_csv(r'D:\研二寒假\课题\课题数据集\鲍鱼数据集/鲍鱼train_分层.csv', error_bad_lines=False, lineterminator="\n", encoding="gbk"))

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
    vae = VAE().to(device)
    print("=====> 构建D")
    D = Discriminator(1).to(device)
    print("=====> 构建C")
    C = Classifier(1).to(device)   #这个是一个标签分类器
    criterion = nn.BCELoss().to(device)   #对一个batch里面的数据做二元交叉熵并且求平均
    MSECriterion = nn.MSELoss().to(device)   #对一个banch里面的数据求均方误差损失并求平均

    print("=====> Setup optimizer")
    optimizerD = optim.Adam(D.parameters(), lr=0.00005)
    optimizerC = optim.Adam(C.parameters(), lr=0.00005)
    optimizerVAE = optim.Adam(vae.parameters(), lr=0.00005)
    err_d = []
    err_c = []
    err_g = []
    for epoch in range(nepoch):
        #定义三个用来存放D,C,G误差的列表：

        for i, (data,label) in enumerate(dataloader, 0):   #data：（200，10）因为banchsize = 200(dataloader),开始索引为0
            # 先处理一下数据
            data = data.type(torch.float32).to(device)
            label = label.type(torch.float32).to(device)   #改变数据类型，label不用再进行编码了，因为分类器是回归的
            batch_size = data.shape[0]
            #生成要生成的目标值y
            y_target = np.random.randint(15, 29, batch_size)  # 生成600-1000内的batch_size个目标值
            y_target = y_target.reshape((-1, 1))
            y_target = Y_scaler.transform(y_target)
            y_target = torch.from_numpy(y_target)
            y_target = y_target.type(torch.float32).to(device)

            # 先训练C
            output = C(data)  #data：（128,10）
            errC = MSECriterion(output, label)   #均方误差
            C.zero_grad()   #初始化梯度
            errC.backward()   #梯度反向传播
            optimizerC.step()   #参数更新
            # 再训练D
            output = D(data)
            output = output.reshape(-1)
            real_label = torch.ones(batch_size).to(device)   # 定义真实的图片label为1
            fake_label = torch.zeros(batch_size).to(device)  # 定义假的图片的label为0
            errD_real = criterion(output, real_label)   #得到判别真实样本的误差


            ###########################这里可以改：把解码器的输入改为噪声（五维）+标签编码（两维）
            z = torch.randn(batch_size, nz).to(device)
            z = torch.cat([z,y_target],1)
            fake_data = vae.decoder(z)   #输出的是（128,10）
            output = D(fake_data)   #（128,1）
            output = output.reshape(-1)
            errD_fake = criterion(output, fake_label)   #得到判别生成样本的误差

            errD = errD_real+errD_fake    #得到判别真实样本和生成样本的误差和
            D.zero_grad()   #初始化判别器D的梯度值
            errD.backward()
            optimizerD.step()   #更新判别器D的参数值

            #更新vae的时候用到的都是生成的样本：
            # 更新VAE(G)1：生成器
            z,mean,logstd = vae.encoder(data)
            #这里要先生成要生成的目标标签：

            z = z.type(torch.float32).to(device)
            z = torch.cat([z,y_target],1)   #把要生成的目标标签值和噪声拼接起来
            recon_data = vae.decoder(z)
            vae_loss1 = loss_function(recon_data,data,mean,logstd)

            # 更新VAE(G)2：判别器
            output = D(recon_data)
            output = output.reshape(-1)
            real_label = torch.ones(batch_size).to(device)
            vae_loss2 = criterion(output,real_label)   #二元交叉熵

            # 更新VAE(G)3   #分类器
            output = C(recon_data)
            real_label = y_target
            vae_loss3 = MSECriterion(output, real_label)

            vae.zero_grad()   #初始化所有的梯度值
            vae_loss = vae_loss1+vae_loss2+vae_loss3   #三个加到一起作为VAE反向传播的误差
            vae_loss.backward()   #反向传播
            optimizerVAE.step()   #梯度更新

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_C: %.4f Loss_G: %.4f'% (epoch, nepoch, i, len(dataloader),errD.item(),errC.item(),vae_loss.item()))

        err_d.append(errD.item())
        err_c.append(errC.item())
        err_g.append(vae_loss.item())

        #取出单元素张量的元素值并返回该值,精度比较好


import matplotlib.pyplot as plt
x = range(nepoch)
y1 = err_d
y2 = err_c
y3 = err_g

plt.plot(x,y1,color='r')
plt.plot(x,y2,color='black')
plt.plot(x,y3,color='blue')

plt.legend()
plt.show()


torch.save(vae.state_dict(), './CVAE-GAN-VAE_train.pth')
# torch.save(D.state_dict(),'./CVAE-GAN-Discriminator.pth')
# torch.save(C.state_dict(),'./CVAE-GAN-Classifier.pth')
