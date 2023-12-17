from __future__ import division

import timeit

import numpy as np

# PyTorch includes
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import trange

# Custom includes
import osvos.custom_transforms as tr
import osvos.davis_2016 as db
import osvos.vgg_osvos as vo


def class_balanced_cross_entropy_loss(output, label, size_average=True, batch_average=True):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """

    labels = torch.ge(label, 0.5).float()

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero))
    )

    loss_pos = torch.sum(-torch.mul(labels, loss_val))
    loss_neg = torch.sum(-torch.mul(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= np.prod(label.size())
    elif batch_average:
        final_loss /= label.size()[0]

    return final_loss


def run(img, label, test_img_list):
    parent_model_path = "C:\\Uni\\Proj\\vidsegtool\\osvos\\models\\parent_epoch-239.pth"

    nAveGrad = 1  # Average the gradient every nAveGrad iterations
    nEpochs = 200 * nAveGrad  # Number of epochs for training
    snapshot = nEpochs  # Store a model every snapshot epochs
    # parentEpoch = 240

    seed = 0

    # Select which GPU, -1 if CPU
    gpu_id = 0
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    print(device)
    # Network definition
    net = vo.OSVOS(pretrained=0)
    net.load_state_dict(torch.load(parent_model_path, map_location=lambda storage, loc: storage))

    # Logging into Tensorboard
    # log_dir = os.path.join(
    #     save_dir, "runs", datetime.now().strftime("%b%d_%H-%M-%S") + "_" + socket.gethostname() + "-" + seq_name
    # )
    # writer = SummaryWriter(log_dir=log_dir)

    net.to(device)  # PyTorch 0.4.0 style

    # Use the following optimizer
    lr = 1e-8
    wd = 0.0002
    optimizer = optim.SGD(
        [
            {
                "params": [pr[1] for pr in net.stages.named_parameters() if "weight" in pr[0]],
                "weight_decay": wd,
            },
            {"params": [pr[1] for pr in net.stages.named_parameters() if "bias" in pr[0]], "lr": lr * 2},
            {
                "params": [pr[1] for pr in net.side_prep.named_parameters() if "weight" in pr[0]],
                "weight_decay": wd,
            },
            {"params": [pr[1] for pr in net.side_prep.named_parameters() if "bias" in pr[0]], "lr": lr * 2},
            {"params": [pr[1] for pr in net.upscale.named_parameters() if "weight" in pr[0]], "lr": 0},
            {"params": [pr[1] for pr in net.upscale_.named_parameters() if "weight" in pr[0]], "lr": 0},
            {"params": net.fuse.weight, "lr": lr / 100, "weight_decay": wd},
            {"params": net.fuse.bias, "lr": 2 * lr / 100},
        ],
        lr=lr,
        momentum=0.9,
    )

    # Preparation of the data loaders
    # Define augmentation transformations as a composition
    composed_transforms = transforms.Compose(
        [tr.RandomHorizontalFlip(), tr.ScaleNRotate(rots=(-30, 30), scales=(0.75, 1.25)), tr.ToTensor()]
    )
    # Training dataset and its iterator
    db_train = db.DAVIS2016(train=True, img_list=[img], labels=[label], transform=composed_transforms)
    trainloader = DataLoader(db_train, batch_size=1, shuffle=True, num_workers=0)

    # Testing dataset and its iterator
    db_test = db.DAVIS2016(
        train=False, img_list=test_img_list, labels=[None] * len(test_img_list), transform=tr.ToTensor()
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    aveGrad = 0

    print("Start of Online Training")
    start_time = timeit.default_timer()
    # Main Training and Testing Loop
    for epoch in trange(0, nEpochs):
        # One training epoch
        running_loss_tr = 0
        np.random.seed(seed + epoch)
        for ii, sample_batched in enumerate(trainloader):

            inputs, gts = sample_batched["image"], sample_batched["gt"]

            # Forward-Backward of the mini-batch
            inputs.requires_grad_()
            inputs, gts = inputs.to(device), gts.to(device)

            outputs = net.forward(inputs)

            # Compute the fuse loss
            loss = class_balanced_cross_entropy_loss(outputs[-1], gts, size_average=False)
            running_loss_tr += loss.item()  # PyTorch 0.4.0 style

            # Print stuff
            if epoch % (nEpochs // 20) == (nEpochs // 20 - 1):
                running_loss_tr /= num_img_tr
                print("[Epoch: %d, numImages: %5d]" % (epoch + 1, ii + 1))
                print("Loss: %f" % running_loss_tr)

            # Backward the averaged gradient
            loss /= nAveGrad
            loss.backward()
            aveGrad += 1

            # Update the weights once in nAveGrad forward passes
            if aveGrad % nAveGrad == 0:
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

        # Save the model
        # if (epoch % snapshot) == snapshot - 1 and epoch != 0:
        #     torch.save(net.state_dict(), os.path.join(save_dir, seq_name + "_epoch-" + str(epoch) + ".pth"))

    stop_time = timeit.default_timer()
    print("Online training time: " + str(stop_time - start_time))

    test_label_list = []

    print("Testing Network")
    with torch.no_grad():  # PyTorch 0.4.0 style
        # Main Testing Loop
        for ii, sample_batched in enumerate(testloader):

            img = sample_batched["image"]

            # Forward of the mini-batch
            inputs = img.to(device)

            outputs = net.forward(inputs)

            for jj in range(int(inputs.size()[0])):
                pred = np.transpose(outputs[-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
                pred = 1 / (1 + np.exp(-pred))
                pred = np.squeeze(pred)

                test_label_list.append(pred)

    return test_label_list
