import argparse
from pathlib import Path


import torch
import numpy as np
import torch.nn as nn
import torchvision
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import imgaug.augmenters as iaa


def load_model(model_name: str, num_classes: int, pretrained: bool = False):
    models = {
        'resnet18': torchvision.models.resnet18,
        'resnet34': torchvision.models.resnet34,
        'resnet50': torchvision.models.resnet50,
        'resnet101': torchvision.models.resnet101,
        'resnet152': torchvision.models.resnet152,
    }
    if pretrained:
        model = models[model_name](weights='IMAGENET1K_V1')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    else:
        model = models[model_name](num_classes=num_classes)
    return model



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
                        choices=['resnet18', 'resnet34', 'resnet50',
                                 'resnet101', 'resnet152'],
                        default='resnet50')

    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'])
    parser.add_argument('--pretrained', action='store_true')

    parser.add_argument('--train-dataset', type=Path, required=True)
    parser.add_argument('--val-dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)

    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=8)


    parser.add_argument('--optimizer', type=str, default='adam')

    parser.add_argument('--resume-checkpoint', type=Path)

    return parser.parse_args()



def train_loop(
        model,
        start_epoch,
        end_epoch,
        optimizer,
        device,
        criterion,
        train_summary_writer,
        val_summary_writer,
        train_dataloader,
        val_dataloader,
        model_save_dir,
        best_acc=0.0
):
    current_best_acc = best_acc
    for epoch in range(start_epoch, end_epoch):
        train_loss, train_accuracy = train_step(
            model, train_dataloader, optimizer, criterion, device, epoch
        )
        train_summary_writer.add_scalar('loss', train_loss, epoch)
        train_summary_writer.add_scalar('accuracy', train_accuracy, epoch)

        val_loss, val_accuracy = val_step(
            model, val_dataloader, criterion, device, epoch
        )
        val_summary_writer.add_scalar('loss/val', val_loss, epoch)
        val_summary_writer.add_scalar('accuracy/val', val_accuracy, epoch)

        if val_accuracy > current_best_acc:
            current_best_acc = val_accuracy
            torch.save(
                {
                        'accuracy': val_accuracy,
                        'loss': val_loss,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    },
                model_save_dir / 'best_model.pt'
            )

        torch.save(
            {
                'accuracy': val_accuracy,
                'loss': val_loss,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            model_save_dir / f'latest_model.pt'
        )


def train_step(model, dataloader, optimizer, criterion, device, epoch):
    model.train()

    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in tqdm(dataloader, desc=f'Train epoch {epoch}'):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc

def val_step(model, dataloader, criterion, device, epoch):
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in tqdm(dataloader, desc=f'Test epoch {epoch}'):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc


def main() -> None:
    args = parse_args()


    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'

    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)

    # log command line args
    with open(output_dir / 'args.txt', 'w') as f:
        f.write(args.__str__())

    device = torch.device(args.device)

    # here we can add albumentations

    seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 3.0))),
                          iaa.Sometimes(0.5, iaa.LinearContrast((0.75, 1.5))),
                          iaa.AdditiveGaussianNoise(loc=0,
                                                    scale=(0.0, 0.05 * 255),
                                                    per_channel=0.5),
                          iaa.ChangeColorTemperature((1100, 10000),
                                                     from_colorspace='RGB'),
                          iaa.Sometimes(0.5, iaa.Affine(
                              scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                              translate_percent={"x": (-0.2, 0.2),
                                                 "y": (-0.2, 0.2)},
                              rotate=(-25, 25),
                              shear=(-8, 8)))], random_order=True)

    train_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        np.asarray,
        seq.augment_image,
        np.copy,
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])
    # train_transforms = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(args.train_dataset, train_transforms)


    val_dataset = datasets.ImageFolder(args.val_dataset, val_transforms)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )


    model = load_model(args.model, len(train_dataset.classes))
    model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_summary_writer = SummaryWriter(log_dir=log_dir / 'train')
    val_summary_writer = SummaryWriter(log_dir=log_dir / 'val')

    start_epoch = 0
    best_accuracy = 0

    if args.resume_checkpoint:
        print(f'Resume training from {args.resume_checkpoint}')
        checkpoint = torch.load(args.resume_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['accuracy']


    print(f'Start epoch {start_epoch}')
    train_loop(model,
               start_epoch,
               args.epochs,
               optimizer,
               device,
               criterion,
               train_summary_writer,
               val_summary_writer,
               train_dataloader,
               val_dataloader,
               checkpoint_dir,
               best_acc=best_accuracy)


if __name__ == '__main__':
    main()