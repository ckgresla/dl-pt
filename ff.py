# Forward-forward Implementation in Torch
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader


# Util to Load in MNIST Data --> used in Hinton's example
def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):

    # Standard Transformations to the MNIST Dataset
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    # Instantiate Loaders (can call these for randomized batches at train/inference time)
    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


# Function to Add the Label information to Images (mentioned in Hinton's paper, see Section 3.3)
def overlay_y_on_x(x, y):
    """
    x & y are the entirety of a batch (whole training set)
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0 #first 10 values of a training instance's 784 long vector
    x_[range(x.shape[0]), y] = x.max()
    return x_


# Network Class that Instantiates our Custom Layers Class & Implements the Train/Predict Funcs
class Net(torch.nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.layers = []
        # Append Layers w Correct Dimensions for Weight Matrices -- dims is a list of ints (i.e; list[int])
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1]).cuda()]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label) #put current label in iteration on training instance
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)] #sum of squared "goodness" -- take max for pred
            # Compute Goodness for Current Label in Range Iters
            goodness_per_label += [sum(goodness).unsqueeze(1)]

        # Get Goodness Over all Labels 
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)


class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        # self.num_epochs = 1000
        self.num_epochs = 10000 #10x increase in epochs, what do?

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

if __name__ == "__main__":
    # torch.manual_seed(1234)
    torch.manual_seed(42)
    train_loader, test_loader = MNIST_loaders() #loaders load in the entirety of the MNIST Set

    # Instantiate Model + Data
    net = Net([784, 500, 500])
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()

    x_pos = overlay_y_on_x(x, y) #add actual labels to training instances


    # Create Random Label for x_negative
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])

    # Go Forward-forward
    net.train(x_pos, x_neg)

    print('train error:', 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()

    print('test error:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())
