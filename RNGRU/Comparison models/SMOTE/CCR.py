import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd


class Smote(object):
    def __init__(self, N=50, k=5, r=2):
        self.N = N
        self.k = k
        self.r = r
        self.newindex = 0

    def fit(self, samples):
        self.samples = samples
        self.T, self.numattrs = self.samples.shape

        if (self.N < 100):
            np.random.shuffle(self.samples)
            self.T = int(self.N * self.T / 100)
            self.samples = self.samples[0:self.T, :]
            self.N = 100

        if (self.T <= self.k):
            self.k = self.T - 1

        N = int(self.N / 100)
        self.synthetic = np.zeros((self.T * N, self.numattrs))

        neighbors = NearestNeighbors(n_neighbors=self.k + 1,
                                     algorithm='ball_tree',
                                     p=self.r).fit(self.samples)

        for i in range(len(self.samples)):
            nnarray = neighbors.kneighbors(self.samples[i].reshape((1, -1)),
                                           return_distance=False)[0][1:]

            self.__populate(N, i, nnarray)

        return self.synthetic

    def __populate(self, N, i, nnarray):
        for j in range(N):
            nn = random.randint(0, self.k - 1)
            diff = self.samples[nnarray[nn]] - self.samples[i]
            gap = random.uniform(0, 1)
            self.synthetic[self.newindex] = self.samples[i] + gap * diff

            self.newindex += 1
samples = np.array(pd.read_csv(r'D:\Pycharmwenjian\train_分层.csv', error_bad_lines=False, lineterminator="\n", encoding="gbk",header=None))
smote = Smote(N=8.4)
synthetic_points1 = smote.fit(samples)
print(synthetic_points1)
np.savetxt(r'D:\研二寒假\课题\课题数据集\SMOTE\100.csv', synthetic_points1, delimiter = ',')