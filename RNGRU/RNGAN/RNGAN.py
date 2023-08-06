
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

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()

        self.RN_fc = nn.Linear(2 * 7, 1)
        self.GRU_gru = torch.nn.GRU(input_size=7, hidden_size=7, num_layers=3)
        self.Sigmoid = nn.Sigmoid()

        self.generator2 = nn.Sequential(nn.Linear(nz+1,6), nn.ReLU(),nn.Linear(6,7))

        self.λ1 = nn.Sequential(nn.Linear(7,1), nn.Sigmoid())

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
            nn.Linear(7, outputn),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = input
        x = self.fd(x)
        return x


class Predictor(nn.Module):
    def __init__(self,outputn=1):
        super(Predictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(7, 5),
            nn.Linear(5, outputn),
            nn.Sigmoid()
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
    cudnn.benchmark = True
    data = np.array(pd.read_csv(r'D:\Pycharmwenjian\RNGRU\CVAE\train_分层.csv', error_bad_lines=False, lineterminator="\n", encoding="gbk"))
    datax = data[:,:-1]
    datay = data[:,-1]
    datay_f = np.array(datay, copy=True)
    datax_f = np.array(datax, copy=True)

    def finddata(mbz,l):

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
    index = [i for i in range(len(datax))]
    np.random.shuffle(index)
    datax = datax[index, :]
    datay = datay[index].reshape((-1,1))

    from sklearn.preprocessing import MinMaxScaler
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()
    datax = torch.from_numpy(X_scaler.fit_transform(datax))
    datay = torch.from_numpy(Y_scaler.fit_transform(datay))
    dataset = TensorDataset(datax,datay)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batchSize,shuffle=True)
    print("=====> 构建generator")
    generator = generator().to(device)
    print("=====> 构建D")
    dis = Discriminator().to(device)
    print("=====> 构建P")
    pre = Predictor().to(device)
    MSECriterion = nn.MSELoss().to(device)
    print("=====> Setup optimizer")
    optimizergenerator = optim.Adam(generator.parameters(), lr=0.0005)
    optimizerD = optim.Adam(dis.parameters(), lr=0.0005)
    optimizerP = optim.Adam(pre.parameters(), lr=0.0005)

    for epoch in range(nepoch):
        for i, (data,label) in enumerate(dataloader, 0):
            data = data.type(torch.float32).to(device)
            label = label.type(torch.float32).to(device)
            for i in range(200):
                pre_out = pre(data)
                err1 = MSECriterion(pre_out, label)
                pre.zero_grad()
                err1.backward(retain_graph=True)
                optimizerP.step()
            batch_size = data.shape[0]
            mubiaozhi_sc = torch.randint(2, 29, (1,))
            mubiaozhi_sc = torch.reshape(mubiaozhi_sc, (1,1))
            mubiaozhi_sc = torch.from_numpy(Y_scaler.fit_transform(mubiaozhi_sc))
            z = torch.randn(1, nz)
            z = torch.cat([z,mubiaozhi_sc],dim=1).to(torch.float32)
            huigui,jinlin = finddata(mubiaozhi_sc, 18)
            output = generator.forward(torch.from_numpy(huigui).to(torch.float32),torch.from_numpy(jinlin).to(torch.float32),z)  #data：（128,10）

            output11 = dis(output)
            output11 = output11.reshape(-1)
            real_label = torch.ones(batch_size).to(device).reshape(-1)
            fake_label = torch.zeros(batch_size).to(device).reshape(-1)
            errD_fake = MSECriterion(output11, fake_label)

            output12 = dis(data)
            output12 = output12.reshape(-1)
            errD_real = MSECriterion(output12, real_label)

            errD = errD_real + errD_fake
            dis.zero_grad()
            errD.backward(retain_graph=True)
            optimizerD.step()

            output21 = dis(output)
            output21 = output21.reshape(-1)
            real_label = torch.ones(batch_size).to(device).reshape(-1)
            fake_label = torch.zeros(batch_size).to(device).reshape(-1)
            errG_fake = MSECriterion(output21, real_label)

            output22 = dis(data)
            output22 = output12.reshape(-1)
            errG_real = MSECriterion(output22, fake_label)
            mubiaozhi_sc = torch.randint(2, 29, (batch_size,))
            mubiaozhi_sc = torch.reshape(mubiaozhi_sc, (batch_size, 1))
            mubiaozhi_bzh = torch.from_numpy(Y_scaler.fit_transform(mubiaozhi_sc))
            shuxing_bzh = torch.from_numpy(Y_scaler.fit_transform(mubiaozhi_sc))

            z = torch.randn(batch_size, nz)
            z = torch.cat([z, mubiaozhi_sc], dim=1)
            huigui1, jinlin1 = finddata(mubiaozhi_sc, 18)
            outputz = generator.forward(torch.from_numpy(huigui1).to(torch.float32),torch.from_numpy(jinlin1).to(torch.float32), z)
            shuxing_bzh = torch.from_numpy(X_scaler.fit_transform(outputz.detach().numpy()))
            yuce = pre(shuxing_bzh)
            err_P = MSECriterion(yuce.to(torch.float32), mubiaozhi_bzh.to(torch.float32))
            err_G = errG_fake+errG_real+err_P
            generator.zero_grad()
            err_G.backward()
            optimizergenerator.step()

    yangben = torch.empty(1, 8)
    for k in range(200):
        yangben1 = torch.empty(1,8)
        mubiaozhi_cs = torch.randint(25, 30, (1,))
        mubiaozhi_cs = torch.reshape(mubiaozhi_cs, (1, 1))
        mubiaozhi_cs = Y_scaler.fit_transform(mubiaozhi_cs)
        mubiaozhi_cs = torch.from_numpy(mubiaozhi_cs)
        z = torch.randn(1, nz)
        z = torch.cat([z, mubiaozhi_cs], dim=1)
        huigui,jinlin = finddata(mubiaozhi_cs, 18)
        output = generator.forward(torch.from_numpy(huigui).to(torch.float32),
                                    torch.from_numpy(jinlin).to(torch.float32), z.to(torch.float32))
        output = X_scaler.inverse_transform(output.detach().numpy())
        mubiaozhi_cs = Y_scaler.inverse_transform(mubiaozhi_cs.detach().numpy())
        yangben1 = torch.cat((torch.from_numpy(output), torch.from_numpy(mubiaozhi_cs)), dim=1)
        yangben = torch.cat((yangben, yangben1), dim=0)
    yangben = yangben.detach().numpy()
    np.savetxt(r'D:\Pycharmwenjian\RNGRU\shengchengyangben\200.csv', yangben,delimiter=',')