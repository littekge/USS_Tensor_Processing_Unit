import torch
import torchvision
import torchvision.transforms as transforms

if __name__ == "__main__":
    # converts images to tensors normalized around x = 0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # defining training dataset (CIFAR10 image dataset)
    trainset = torchvision.datasets.CIFAR10(root='./Pytorch_Tests/data/', train=True, download=True, transform=transform)

    # loading training data
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)


    # Code to show data
    import matplotlib.pyplot as plt
    import numpy as np

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    plt.show()