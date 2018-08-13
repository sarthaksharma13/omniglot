import torch
import pickle
import torchvision
from torchvision import transforms
import torchvision.datasets as dset
from torchvision import transforms
from mydataset import OmniglotTrain, OmniglotTest
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import Siamese
import time
import numpy as np


import args
# Parse commandline arguements
cmd = args.arguments;

# Check if cuda is available for GPU usage.
cuda = torch.cuda.is_available()


data_transforms = transforms.Compose([
    transforms.RandomAffine(15),
    transforms.ToTensor()
])

# Assuming you have run make_dataset.py as specified.
train_path = 'background'
test_path = 'evaluation'
train_dataset = dset.ImageFolder(root=train_path)
test_dataset = dset.ImageFolder(root=test_path)

way = 20
times = 400

dataSet = OmniglotTrain(train_dataset, transform=data_transforms)
testSet = OmniglotTest(test_dataset, transform=transforms.ToTensor(), times = times, way = way)
testLoader = DataLoader(testSet, batch_size=way, shuffle=False, num_workers=16)

dataLoader = DataLoader(dataSet, batch_size=cmd.trainBatch,\
                        shuffle=False, num_workers=16)




# Get the network architecture
net = Siamese()
# Loss criterion
criterion = torch.nn.BCEWithLogitsLoss(size_average=True)

# Optimizer
if cmd.optMethod == 'adam':
    optimizer = torch.optim.Adam(net.parameters(),lr = cmd.lr )

# To store train loss
train_loss = []
# To store the accuracy
accuracy = []
# Get the network in training mode.
net.train()

# Use GPUs.
if cuda:
    net.cuda()

# Parameters to show, save and test
show_every = 10
save_every = 100
test_every = 100

# Track the loss
loss_val = 0


for batch_id, (img1, img2, label) in enumerate(dataLoader, 1):
    # Max iters 
    if batch_id > cmd.iters:
        break
    # Start time
    batch_start = time.time()

    # If GPU, convert to cuda tensor
    if cuda:
        img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
    else:
        img1, img2, label = Variable(img1), Variable(img2), Variable(label)

    # Zero gradient parameters from previous batch
    optimizer.zero_grad()
    # Forward the image
    output = net.forward(img1, img2)
    # Compute the loss
    loss = criterion(output, label)

    loss_val += loss.data[0]
    # Backprop
    loss.backward()
    # Take the optimizer step
    optimizer.step()
    # For saving, displaying and testing.
    if batch_id % show_every == 0 :
        print('[%d]\tloss:\t%.5f\tTook\t%.2f s'%(batch_id, loss_val/show_every, (time.time() - batch_start)*show_every))
        loss_val = 0
    #if batch_id % save_every == 0:
        #torch.save(net.state_dict(), './model/model-batch-%d.pth'%(batch_id+1,))
    if batch_id % test_every == 0:
        right, error = 0, 0
        for _, (test1, test2) in enumerate(testLoader, 1):
            if cuda:
                test1, test2 = test1.cuda(), test2.cuda()
            test1, test2 = Variable(test1), Variable(test2)
            output = net.forward(test1, test2).data.cpu().numpy()
            pred = np.argmax(output)
            if pred == 0:
                right += 1
            else: error += 1
        print('*'*70)
        print('[%d]\tright:\t%d\terror:\t%d\tprecision:\t%f'%(batch_id, right, error, right*1.0/(right+error)))
        print('*'*70)
        accuracy.append(right*100.0/(right+error))

    train_loss.append(loss_val)


with open('train_loss', 'wb') as f:
    pickle.dump(train_loss, f)


fig, ax = plt.subplots(1)
ax.plot(range(len(train_loss)), train_loss, 'r', label = 'loss')
ax.legend()
plt.ylabel('Loss')
plt.xlabel('Batch #')
fig.savefig(os.path.join('./loss_val'))

fig, ax = plt.subplots(1)
ax.plot(range(len(accuracy)), accuracy, 'b', label = 'loss')
ax.legend()
plt.ylabel('Accuracy ')
plt.xlabel('Batch #')
fig.savefig(os.path.join('./accuracy'))
