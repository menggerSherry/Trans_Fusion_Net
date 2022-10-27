import torch

def calIOU(output, mask):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.data.cpu().numpy()

    output_ = output > 0.5
    mask_ = mask > 0.5
    intersection = (output_ & mask_).sum()
    union = (output_ | mask_).sum()

    return (intersection + smooth) / (union + smooth)


def calDICE(output, mask):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.data.cpu().numpy()

    intersection = (output * mask).sum()

    return (2 * intersection + smooth) / (output.sum() + mask.sum() + smooth)

def calSEN(output, mask):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.data.cpu().numpy()

    intersection = (output * mask).sum()

    return (intersection + smooth) / (mask.sum() + smooth)

def calVOE(output, mask):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.data.cpu().numpy()

    intersection = mask.sum()

    return (2 * (output.sum() - intersection) + smooth) / (output.sum() + intersection + smooth)


def calRVD(output, mask):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.data.cpu().numpy()

    intersection = mask.sum()

    return (output.sum() + smooth) / (intersection - 1 + smooth)


def calPrecision(output, mask):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.data.cpu().numpy()

    intersection = (output * mask).sum()

    return (intersection + smooth) / (output.sum() + smooth)


def calRecall(output, mask):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.data.cpu().numpy()

    intersection = (output * mask).sum()

    return (intersection + smooth) / (mask.sum() + smooth)

