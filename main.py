import argparse

from munkres import Munkres
import torch
import torch.utils.data
from torchvision import datasets, transforms

from vade import VaDE, lossfun

N_CLASSES = 10


def train(model, data_loader, optimizer, device, epoch):
    model.train()

    total_loss = 0
    for x, _ in data_loader:
        x = x.to(device).view(-1, 784)
        recon_x, mu, logvar = model(x)
        loss = lossfun(model, x, recon_x, mu, logvar)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch {:>3}: ELBO = {:.4f}'.format(
        epoch, -total_loss / len(data_loader)))


def test(model, data_loader, device, epoch):
    model.eval()

    gain = torch.zeros((N_CLASSES, N_CLASSES), dtype=torch.int, device=device)
    with torch.no_grad():
        for xs, ts in data_loader:
            xs, ts = xs.to(device).view(-1, 784), ts.to(device)
            ys = model.classify(xs)
            for t, y in zip(ts, ys):
                gain[t, y] += 1
        cost = (torch.max(gain) - gain).cpu().numpy()
        assign = Munkres().compute(cost)
        acc = torch.sum(gain[tuple(zip(*assign))]).float() / torch.sum(gain)

    print('Epoch {:>3}: ACC = {:.2%}'.format(epoch, acc))


def main():
    parser = argparse.ArgumentParser(
        description='Train VaDE with MNIST dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', '-e',
                        help='Number of epochs.',
                        type=int, default=100)
    parser.add_argument('--gpu', '-g',
                        help='GPU id. (Negative number indicates CPU)',
                        type=int, default=-1)
    parser.add_argument('--learning-rate', '-l',
                        help='Learning Rate.',
                        type=float, default=0.002)
    parser.add_argument('--batch-size', '-b',
                        help='Batch size.',
                        type=int, default=100)
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

    model = VaDE(10, 784, 10).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # LR decreases every 10 epochs with a decay rate of 0.9
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.9)

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, device, epoch)
        test(model, test_loader, device, epoch)
        lr_scheduler.step()


if __name__ == '__main__':
    main()
