import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from Layer import MyIris

iris = datasets.load_iris()
xtrain, xtest, ytrain, ytest = train_test_split(
    iris.data, iris.target, test_size=0.5)

xtrain = torch.from_numpy(xtrain).type('torch.FloatTensor')
ytrain = torch.from_numpy(ytrain).type('torch.LongTensor')
xtest = torch.from_numpy(xtest).type('torch.FloatTensor')
ytest = torch.from_numpy(ytest).type('torch.LongTensor')

model = MyIris()
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

DATA_SIZE = 75
BATCH_SIZE = 25
model.train()
for epoch in range(1000):
    idx = np.random.permutation(DATA_SIZE)
    for i in range(0, DATA_SIZE, BATCH_SIZE):
        xtm = xtrain[idx[i:(i+BATCH_SIZE) if (i + BATCH_SIZE)
                         < DATA_SIZE else DATA_SIZE]]
        ytm = ytrain[idx[i:(i+BATCH_SIZE) if (i + BATCH_SIZE)
                         < DATA_SIZE else DATA_SIZE]]
        output = model(xtm)
        loss = criterion(output, ytm)
        print(epoch, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'myiris.model')
model.load_state_dict(torch.load('myiris.model'))
model.eval()
with torch.no_grad():
    output1 = model(xtest)
    ans = torch.argmax(output1, 1)
    print(((ytest == ans).sum().float() / len(ans)).item())
