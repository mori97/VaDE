import argparse

from sklearn.mixture import GaussianMixture
import torch
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms

from vade import AutoEncoderForPretrain, VaDE

N_CLASSES = 10


def train(model, data_loader, optimizer, device, epoch):
    model.train()

    total_loss = 0
    for x, _ in data_loader:
        x = x.to(device).view(-1, 784)
        recon_x = model(x)
        loss = F.mse_loss(recon_x, x)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch {:>3}: Train Loss = {:.4f}'.format(
        epoch, total_loss / len(data_loader)))


def test(model, data_loader, device, epoch):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device).view(-1, 784)
            recon_x = model(x)
            loss = F.mse_loss(recon_x, x)
            total_loss += loss.item()

    print('Epoch {:>3}:  Test Loss = {:.4f}'.format(
        epoch, total_loss / len(data_loader)))


def main():
    parser = argparse.ArgumentParser(
        description='Train VaDE with MNIST dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', '-e',
                        help='Number of epochs.',
                        type=int, default=20)
    parser.add_argument('--gpu', '-g',
                        help='GPU id. (Negative number indicates CPU)',
                        type=int, default=-1)
    parser.add_argument('--learning-rate', '-l',
                        help='Learning Rate.',
                        type=float, default=0.001)
    parser.add_argument('--batch-size', '-b',
                        help='Batch size.',
                        type=int, default=128)
    parser.add_argument('--out', '-o',
                        help='Output path.',
                        type=str, default='./vade_parameter.pth')
    args = parser.parse_args()

    if_use_cuda = torch.cuda.is_available() and args.device
    device = torch.device('cuda:{}'.format(args.gpu) if if_use_cuda else 'cpu')

    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                   transform=transforms.ToTensor())
    test_dataset = datasets.MNIST('./data', train=False, download=True,
                                  transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=if_use_cuda)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=if_use_cuda)

    pretrain_model = AutoEncoderForPretrain(784, 10).to(device)

    optimizer = torch.optim.Adam(pretrain_model.parameters(),
                                 lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        train(pretrain_model, train_loader, optimizer, device, epoch)
        test(pretrain_model, test_loader, device, epoch)

    with torch.no_grad():
        x = torch.cat([train_dataset[i][0] for i in range(len(train_dataset))])
        x = x.view(-1, 784).to(device)
        z = pretrain_model.encode(x).cpu()

    pretrain_model = pretrain_model.cpu()
    state_dict = pretrain_model.state_dict()

    gmm = GaussianMixture(n_components=10, covariance_type='diag')
    gmm.fit(z)

    model = VaDE(N_CLASSES, 784, 10)
    model.load_state_dict(state_dict, strict=False)
    model._pi.data = torch.log(torch.from_numpy(gmm.weights_)).float()
    model.mu.data = torch.from_numpy(gmm.means_).float()
    model.logvar.data = torch.log(torch.from_numpy(gmm.covariances_)).float()

    torch.save(model.state_dict(), args.out)


if __name__ == '__main__':
    main()
