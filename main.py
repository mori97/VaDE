import argparse

from munkres import Munkres
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from vade import VaDE, lossfun

N_CLASSES = 10


def train(model, data_loader, optimizer, device, epoch, writer):
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

    writer.add_scalar('Loss/train', total_loss / len(data_loader), epoch)


def test(model, data_loader, device, epoch, writer):
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

    writer.add_scalar('Acc/test', acc, epoch)
    writer.flush()


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
    parser.add_argument('--pretrain', '-p',
                        help='Load parameters from pretrained model.',
                        type=str, default=None)
    args = parser.parse_args()

    if_use_cuda = torch.cuda.is_available() and args.gpu >= 0
    device = torch.device('cuda:{}'.format(args.gpu) if if_use_cuda else 'cpu')

    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=if_use_cuda)

    model = VaDE(N_CLASSES, 784, 10)
    if args.pretrain:
        model.load_state_dict(torch.load(args.pretrain))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # LR decreases every 10 epochs with a decay rate of 0.9
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.9)

    # TensorBoard
    writer = SummaryWriter()

    for epoch in range(1, args.epochs + 1):
        train(model, data_loader, optimizer, device, epoch, writer)
        test(model, data_loader, device, epoch, writer)
        lr_scheduler.step()

    writer.close()


if __name__ == '__main__':
    main()
