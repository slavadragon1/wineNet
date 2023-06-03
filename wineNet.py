# %% [markdown]
# ## **Import**
# 
# ---

# %%
import torch
import random
import numpy as np

import sklearn.datasets
wine = sklearn.datasets.load_wine()

# %% [markdown]
# ## **Preparing datasets**
# 
# ---

# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    wine.data[:, :2],  #first 2 columns
    wine.target,
    test_size = 0.3,
    shuffle=True
)
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
y_train

# %% [markdown]
# ## **Model**
# 
# ---

# %%
class WineNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons) -> None:
        super(WineNet, self).__init__()

        self.fc1 = torch.nn.Linear(2, n_hidden_neurons)
        self.activ1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.activ2 = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(n_hidden_neurons, 3)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ1(x)
        x = self.fc2(x)
        x = self.activ2(x)
        x = self.fc3(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x

wine_net = WineNet(5)

# %% [markdown]
# ### **Loss and optimizer**
# 
# ---

# %%
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(wine_net.parameters(), lr=0.001)

# %% [markdown]
# ### **Learning**
# 
# ---

# %%
batch_size = 10

for epoch in range(5000):
    order = np.random.permutation(len(x_train))
    for start_index in range(0, len(x_train), batch_size): #куски по 10 
        optimizer.zero_grad()

        batch_indexes = order[start_index:start_index+batch_size]

        x_batch = x_train[batch_indexes]
        y_batch = y_train[batch_indexes]

        predicts = wine_net.forward(x_batch)

        loss_value = loss(predicts, y_batch)
        loss_value.backward()

        optimizer.step()

    if epoch % 100 == 0:
        test_preds = wine_net.forward(x_test)
        test_preds = test_preds.argmax(dim=1)
        print((test_preds == y_test).float().mean())

# %%


import matplotlib.pyplot as plt
%matplotlib inline

plt.rcParams['figure.figsize'] = (5,4)

n_classes = 3
plot_colors = ['g', 'orange', 'black']
plot_step = 0.02

x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1

xx, yy =  torch.meshgrid(torch.arange(x_min, x_max, plot_step),
                         torch.arange(y_min, y_max, plot_step))

preds = wine_net.inference(
    torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1))

preds_class = preds.data.numpy().argmax(axis=1)
preds_class = preds_class.reshape(xx.shape)
plt.contourf(xx, yy, preds_class, cmap='Accent')

for i, color in zip(range(n_classes), plot_colors):
    indexes = np.where(y_train == i)
    plt.scatter(x_train[indexes, 0], 
                x_train[indexes, 1], 
                c=color, 
                label=wine.target_names[i],
                cmap='Accent')
    plt.xlabel(wine.feature_names[0])

    plt.ylabel(wine.feature_names[1])
    plt.legend()




