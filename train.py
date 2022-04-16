import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model import shufflenet_v2_x1_0
#from shufflenet_cabm import shufflenet_v2_x1_0
from my_dataset import MyDataSet
from utils import read_data, train_one_epoch, evaluate

import matplotlib.pyplot as plt


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    #train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    train_images_path, train_images_label, val_images_path, val_images_label = read_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    #nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    #print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=4,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=4,
                                             collate_fn=val_dataset.collate_fn)

    # 如果存在预训练权重则载入
    model = shufflenet_v2_x1_0(num_classes=args.num_classes).to(device)
    #model = shufflenet_v2_x1_0(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # =========初始化横纵坐标列表=====
    mean_loss_list = []
    accurate_list = []
    epoch_list = []
    learning_rate_list = []
    # ================================

    for epoch in range(args.epochs):

        # ========向epoch列表中添加数值===
        epoch_list.append(epoch + 1)
        # ===============================

        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        acc = evaluate(model=model,
                       data_loader=val_loader,
                       device=device)

        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model, "./weights/model-{}.pth".format(epoch))

        # =========向loss和准确率列表添加每个epoch的数值===
        mean_loss_list.append(mean_loss)
        accurate_list.append(acc)
        learning_rate_list.append(optimizer.param_groups[0]["lr"])
        # ===============================================

    # ==========画loss图================
    plt.plot(epoch_list, mean_loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Train loss vs.epochs')
    plt.savefig("loss_plt/loss1.png")
    plt.show()
    # ==================================

    # ===========画准确率图==============
    plt.plot(epoch_list, accurate_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Val accuracy vs.epochs')
    plt.savefig("accuracy_plt/accuracy1.png")
    plt.show()
    # ==================================

    # ===========画学习率图==============
    plt.plot(epoch_list, learning_rate_list)
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.title('Learning_rate vs.epochs')
    plt.savefig("learning_rate_plt/learning_rate1.png")
    plt.show()
    # ==================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgzG
    parser.add_argument('--data-path', type=str,default=r".\data_set")
    # shufflenetv2_x1.0 官方权重下载地址
    # https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
    parser.add_argument('--weights', type=str, default=r'.\pretrain_model\shufflenetv2_x1.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
