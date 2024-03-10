import torch
from components.Dataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
from components.Model import SentimentClassifier
import torch
from transformers import AdamW
from torch.nn import BCEWithLogitsLoss
from components.Logger import logger
import torch.nn as nn
import os

def _save_model(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

def train(args):
    model = SentimentClassifier(n_classes=1)

    if torch.cuda.device_count() > 1:
        logger.info("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    optimizer= AdamW(model.parameters(), lr=5e-5)
    data_loader = DataLoader(CustomDataset(args.data_dir))

    optimizer = AdamW(model.parameters(), lr=5e-5)

    loss_fn = BCEWithLogitsLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(args.epochs):
        for _, (data, labels) in enumerate(data_loader):
            optimizer.zero_grad()

            iid, mask, tiid, tmask = data 
            iid = iid.to(device)
            mask = mask.to(device)
            tiid = tiid.to(device)
            tmask = tmask.to(device)

            outputs = model(iid, mask, tiid, tmask)
            labels = labels.to(device)
            labels = labels.view(-1, 1)
            loss = loss_fn(outputs, labels.float())  
            loss.backward()
            optimizer.step()

            logger.info(f"Epoch {epoch+1}, Loss: {loss.item()}")
            
    return _save_model(model, args.model_dir)
    