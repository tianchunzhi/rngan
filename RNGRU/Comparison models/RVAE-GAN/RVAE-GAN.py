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

class VAE(nn.Module):   
    def __init__(self):
        super(VAE, self).__init__()   
        self.encoder_fc1 = nn.Linear(7,nz) 
        self.encoder_fc2 = nn.Linear(7,nz)
        self.decoder_fc = nn.Linear(nz+1,7)
        self.Sigmoid = nn.Sigmoid()
    def noise_reparameterize(self,mean,logvar):
        eps = torch.randn(mean.shape).to(device)   
        z = mean + eps * torch.exp(logvar)
        return z
    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output
    def encoder(self,x):
        x = x
        mean = self.encoder_fc1(x)
        logstd = self.encoder_fc2(x)
        z = self.noise_reparameterize(mean, logstd)  
        return z,mean,logstd


    def decoder(self,z):
        out3 = self.decoder_fc(z)
        return out3

class Discriminator(nn.Module):
    def __init__(self,outputn=1):
        super(Discriminator, self).__init__()
        self.fd = nn.Sequential(
            nn.Linear(7, outputn),   
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
            nn.Linear(7, 5),  
            nn.Linear(5, outputn)
        )

    def forward(self, input):
        x = input
        x = self.fc(x)
        return x

def loss_function(recon_x,x,mean,logstd):
    MSE = MSECriterion(recon_x,x)  
    var = torch.pow(torch.exp(logstd),2) 
    KLD = -0.5 * torch.sum(1+torch.log(var)-torch.pow(mean,2)-var)  
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
    cudnn.benchmark = True

    data = np.array(pd.read_csv(r'D:\研二寒假\课题\课题数据集\train_分层.csv', error_bad_lines=False, lineterminator="\n", encoding="gbk"))

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
    vae = VAE().to(device)
    print("=====> 构建D")
    D = Discriminator(1).to(device)
    print("=====> 构建C")
    C = Classifier(1).to(device) 
    criterion = nn.BCELoss().to(device)   
    MSECriterion = nn.MSELoss().to(device)   

    print("=====> Setup optimizer")
    optimizerD = optim.Adam(D.parameters(), lr=0.00005)
    optimizerC = optim.Adam(C.parameters(), lr=0.00005)
    optimizerVAE = optim.Adam(vae.parameters(), lr=0.00005)
    err_d = []
    err_c = []
    err_g = []
    for epoch in range(nepoch):
        for i, (data,label) in enumerate(dataloader, 0):  
            data = data.type(torch.float32).to(device)
            label = label.type(torch.float32).to(device)   
            batch_size = data.shape[0]
            y_target = np.random.randint(15, 29, batch_size)  
            y_target = y_target.reshape((-1, 1))
            y_target = Y_scaler.transform(y_target)
            y_target = torch.from_numpy(y_target)
            y_target = y_target.type(torch.float32).to(device)

            output = C(data) 
            errC = MSECriterion(output, label)   
            C.zero_grad()  
            errC.backward()   
            optimizerC.step()  
            output = D(data)
            output = output.reshape(-1)
            real_label = torch.ones(batch_size).to(device)   
            fake_label = torch.zeros(batch_size).to(device)  
            errD_real = criterion(output, real_label)   

            z = torch.randn(batch_size, nz).to(device)
            z = torch.cat([z,y_target],1)
            fake_data = vae.decoder(z)   
            output = D(fake_data)   
            output = output.reshape(-1)
            errD_fake = criterion(output, fake_label)  

            errD = errD_real+errD_fake   
            D.zero_grad()   
            errD.backward()
            optimizerD.step()   
            z,mean,logstd = vae.encoder(data)

            z = z.type(torch.float32).to(device)
            z = torch.cat([z,y_target],1)  
            recon_data = vae.decoder(z)
            vae_loss1 = loss_function(recon_data,data,mean,logstd)

            output = D(recon_data)
            output = output.reshape(-1)
            real_label = torch.ones(batch_size).to(device)
            vae_loss2 = criterion(output,real_label)   

            output = C(recon_data)
            real_label = y_target
            vae_loss3 = MSECriterion(output, real_label)

            vae.zero_grad()   
            vae_loss = vae_loss1+vae_loss2+vae_loss3   
            vae_loss.backward()  
            optimizerVAE.step()  
        err_d.append(errD.item())
        err_c.append(errC.item())
        err_g.append(vae_loss.item())

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


torch.save(vae.state_dict(), './a_train.pth')
