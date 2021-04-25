from torch import nn
import torch

torch.manual_seed(1)


class LogisticRegression(nn.Module):
    def __init__(self,
                 dim_in, dim_out):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        # reshape / flatten tensors (ex. MNIST 28*28 -> 784)
        flatten_dim = 1
        for ix in range(1, len(x.shape)):
            flatten_dim *= x.shape[ix]

        # Forward loop
        x = x.reshape(-1, flatten_dim)
        z = self.fc(x)
        return z


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, hidden1, hidden2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim_in, hidden1)
        # self.fc1 = nn.DataParallel(self.fc1)
        # torch.nn.init.zeros_(self.fc1.weight)
        self.hidden1 = nn.Linear(hidden1, hidden2)
        # self.hidden1 = nn.DataParallel(self.hidden1)
        # torch.nn.init.zeros_(self.hidden1.weight)
        self.fc2 = nn.Linear(hidden2, dim_out)
        # self.fc2 = nn.DataParallel(self.fc2)
        # torch.nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        flatten_dim = 1
        for ix in range(1, len(x.shape)):
            flatten_dim *= x.shape[ix]
        x = x.reshape(x.shape[0], flatten_dim)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.hidden1(x))
        z = self.fc2(x)
        # z = nn.functional.log_softmax(z, dim=1)
        return z
