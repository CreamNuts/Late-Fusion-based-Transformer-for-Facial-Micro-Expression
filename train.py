from typing import Dict

import torch
from torch import distributed as dist
from tqdm import tqdm

from meter import SubjMeter
from utils import Visualizer


def trainval(
    epochs,
    model,
    optimizer,
    criterion,
    trainloader,
    valloader,
    writer,
    local_rank=0,
    distributed=False,
    world_size=1,
    visualize=False,
) -> Dict:
    subjmeter = SubjMeter(writer)
    if local_rank == 0:
        pbar = tqdm(range(epochs))
    for epoch in range(epochs):
        if hasattr(trainloader.sampler, "set_epoch"):
            trainloader.sampler.set_epoch(epoch)
        # Train
        model.train()
        for i, (data, label) in enumerate(trainloader, 1):
            optimizer.zero_grad()
            data = data.cuda()
            label = label.cuda()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            predict = torch.max(output, dim=1)[1]
            # Record train loss and metrics
            torch.cuda.synchronize()
            if distributed:
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= world_size
                dist_label = [
                    torch.empty_like(label, dtype=label.dtype).cuda() for _ in range(world_size)
                ]
                dist.all_gather(dist_label, label)
                label = torch.cat(dist_label)
                dist_predict = [
                    torch.empty_like(predict, dtype=predict.dtype).cuda()
                    for _ in range(world_size)
                ]
                dist.all_gather(dist_predict, predict)
                predict = torch.cat(dist_predict)
            subjmeter.train.loss -= (subjmeter.train.loss - loss.detach().cpu()) / i
            subjmeter.train(
                label.detach().cpu().numpy(), predict.detach().cpu().numpy(), output.shape[1]
            )
        # Test
        model.eval()
        for i, (data, label) in enumerate(valloader, 1):
            with torch.no_grad():
                data = data.cuda()
                label = label.cuda()
                output = model(data)
                loss = criterion(output, label)
                predict = torch.max(output, dim=1)[1]
                # Record val loss and metrics
                subjmeter.val.loss -= (subjmeter.val.loss - loss.detach().cpu()) / i
                subjmeter.val(
                    label.detach().cpu().numpy(), predict.detach().cpu().numpy(), output.shape[1]
                )
        if local_rank == 0:
            pbar.update()
        if writer is not None and visualize:
            visualizer = Visualizer(trainloader.dataset.int2str)
            train_sample = trainloader.dataset.get_sample()
            train_predict = visualizer.make_videos(model, train_sample)
            val_sample = valloader.dataset.get_sample()
            val_predict = visualizer.make_videos(model, val_sample)
            writer.add_video("Train Predict", train_predict, epoch, fps=4)
            writer.add_video("Val Predict", val_predict, epoch, fps=4)
        subjmeter.reset()
    return subjmeter
