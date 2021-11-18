from typing import Dict

import torch
from tqdm import trange

from meter import SubjMeter


def trainval(
    device: torch.device, epochs, model, optimizer, criterion, trainloader, valloader, writer
) -> Dict:
    subjmeter = SubjMeter(writer)
    for epoch in trange(epochs):
        # Train
        model.train()
        for i, (data, label) in enumerate(trainloader, 1):
            optimizer.zero_grad()
            output = model(data.to(device))
            loss = criterion(output, label.to(device))
            loss.backward()
            optimizer.step()
            predict = torch.max(output, dim=1)[1].detach().cpu()
            # Record train loss and metrics
            subjmeter.train.loss -= (subjmeter.train.loss - loss.detach().cpu()) / i
            subjmeter.train(label.numpy(), predict.numpy(), output.shape[1])
        # Test
        model.eval()
        for i, (data, label) in enumerate(valloader, 1):
            with torch.no_grad():
                output = model(data.to(device))
                loss = criterion(output, label.to(device)).cpu()
                predict = torch.max(output, dim=1)[1].detach().cpu()
                # Record val loss and metrics
                subjmeter.val.loss -= (subjmeter.val.loss - loss.detach().cpu()) / i
                subjmeter.val(label.numpy(), predict.numpy(), output.shape[1])

        # Writer for TensorBoard
        subjmeter.reset()
    return subjmeter
