
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
        self.decoder_fc = nn.Sequential(nn.Linear(nz+2,6), nn.ReLU(),nn.Linear(6,8))   
    #生成器前向传播
    def forward(self, x):
        output = self.decoder_fc(x)
        return output

class Discriminator(nn.Module):
    def __init__(self,outputn=1):
        super(Discriminator, self).__init__()
        self.fd = nn.Sequential(
            nn.Linear(8, outputn),  
            nn.Sigmoid()  
        )

    def forward(self, input):
        x = input
        x = self.fd(x)
        return x


class Classifier(nn.Module):
    def __init__(self,outputn=1):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(8, 5),  
            nn.Linear(5, outputn),
            nn.Sigmoid()   
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
    print("=====> 构建VAE")
    G = generator().to(device)
    print("=====> 构建D")
    D = Discriminator(1).to(device)
    print("=====> 构建C")
    C = Classifier(1).to(device)   
    criterion = nn.BCELoss().to(device)  
    MSECriterion = nn.MSELoss().to(device)  

    print("=====> Setup optimizer")
    optimizerD = optim.Adam(D.parameters(), lr=0.0005)
    optimizerC = optim.Adam(C.parameters(), lr=0.0005)
    optimizerG = optim.Adam(G.parameters(), lr=0.0005)
    err_d = []
    err_c = []
    err_g = []
    for epoch in range(nepoch):

        for i, (data,label) in enumerate(dataloader, 0):   
            data = data.type(torch.float32).to(device)
            label = label.type(torch.float32).to(device)   
            batch_size = data.shape[0]
            y_target = np.ones(batch_size)
            y_target = torch.from_numpy(y_target)
            y_target = y_target.type(torch.float32).to(device)
            y_target_one = y_target.type(torch.long)  # 改变数据类型
            label_onehot_f = torch.zeros((batch_size, 2)).to(device)  # (128,2)标签种类为2
            label_onehot_f[torch.arange(batch_size).type(torch.long), y_target_one] = 1  # 相当于对标签进行独热编码
            output = C(data) 
            for e in range(batch_size):
                if output[e] >= 0.5:
                    output[e]=1
                elif output[e] < 0.5:
                    output[e]=0
            errC = MSECriterion(output, label)   
            C.zero_grad()  
            errC.backward()   
            optimizerC.step()   
            output = D(data)
            output = output.reshape(-1)   
            for f in range(batch_size):
                if output[f] >= 0.5:
                    output[f]=1
                elif output[f] < 0.5:
                    output[f]=0

            real_label = torch.ones(batch_size).to(device)   
            fake_label = torch.zeros(batch_size).to(device)  
            errD_real = criterion(output, real_label)   
            z = torch.randn(batch_size, nz).to(device)
            z = torch.cat([z,label_onehot_f],1)
            fake_data = G.forward(z)   
            output = D(fake_data)  
            output = output.reshape(-1)
            for m in range(batch_size):
                if output[m] >= 0.5:
                    output[m]=1
                elif output[m] < 0.5:
                    output[m]=0

            errD_fake = criterion(output, fake_label)   

            errD = errD_real+errD_fake   
            D.zero_grad()  
            errD.backward()
            optimizerD.step()   

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_C: %.4f '% (epoch, nepoch, i, len(dataloader),errD.item(),errC.item()))

        err_d.append(errD.item())
        err_c.append(errC.item())

torch.save(G.state_dict(), './CVAE-GAN-VAE_train.pth')
