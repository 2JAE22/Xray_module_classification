import torch


def Xray_collate_fn(batch):
    batch = torch.utils.data.dataloader.default_collate(batch)
    return batch
