import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.Linear(5, 1)
    
    def forward(self, x):
        x.requires_grad_(True)
        x = self.layer(x)
        return x

x = torch.randint(0, 100, (10, 100)).cuda()

model_1 = nn.Embedding(100, 5).cuda()
model_1.requires_grad_(False)

model_2 = nn.Linear(5, 5).cuda()
model_2.requires_grad_(False)

model_3 = Model().cuda()
model_3.requires_grad_(True)

optim = torch.optim.Adam(model_3.parameters(), lr=10000)

with torch.no_grad():
    TEST_VALUE = torch.rand(10, 5).cuda()
    T = model_3(TEST_VALUE).clone().detach()

def forward(x):
    t = model_1(x) # x is not grad
    f = model_2(t) # t is not grad
    return model_3(f) # return is grad

y = forward(x)
loss = y.mean()
loss.backward()
optim.step()
print("Difference", model_3(TEST_VALUE) - T)
print(model_3.parameters().__next__().grad)

optim.zero_grad()
y = model_1(x)
y = model_2(y)
y.requires_grad_(True)
y = model_3(y)
loss = y.mean()
loss.backward()
optim.step()
print("Difference", model_3(TEST_VALUE) - T)
print(model_3.parameters().__next__().grad)