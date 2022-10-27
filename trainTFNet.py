from numpy.core.shape_base import block
import torch
from torch.utils import data
import argparse
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from glob import glob
import os
from configs import all_config
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from utils import AverageMeter, iou_score, Dataset,EdgeDataset
import losses

# from models.SEUNet import SEUNet
# from models.UNet import UNet
# from models.Transformer import Transformer
# from models.UNet3plus import *
#from models.VesselTNet import VTTransNet,CONFIGS,MultiScaleDiscriminator
#from models.VesselTNet import *
from models.TransSimeanNet import *
#from models.TransSimeanNet_v2 import *
from metrics import calIOU, calDICE, calVOE, calRVD, calPrecision, calRecall, calSEN
# if cuda is avaliable then use it,else use cpu
cuda_flag = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_flag else "cpu")


# parser argument
parser = argparse.ArgumentParser(description = 'Vessel$Tumor')

parser.add_argument('--mode', default='train', type=str, choices=['train', 'test','retrain'])
parser.add_argument('--block', default='VesselTBlock2', type=str)
parser.add_argument('--train_mode', default='train', type=str, choices=['train', 'test','retrain','gan_train'])
parser.add_argument('--fuc', default='vessel', type=str, choices=['pretrain_vessel','vessel', 'tumor','pretrain_tumor'])
parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
parser.add_argument('--decay_epoch', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=2, type=int, help='batch_size(default:16)')
parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate,0.001')
parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--model', default='Transformer', type=str, choices=['SEUNet', 'UNet', 'Transformer'])

args = parser.parse_args()


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0 
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, x): 
        return 1.0 - max(0, x + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def test(data_generator, model,alpha1,alpha2):
    log = {'loss': AverageMeter(),
     'iou': AverageMeter(),
     'dice': AverageMeter(),
    'voe': AverageMeter(),
    'rvd': AverageMeter(),
    'sen':AverageMeter(),
    'precision': AverageMeter(),
    'recall': AverageMeter()}

    model.eval()

    with torch.no_grad():
        for image, mask,edge in data_generator:
            image = image.to(device)
            mask = mask.to(device)
            edge = edge.to(device)
            loss_use = losses.TotalLoss().to(device)
            output,pred_edge = model(image)
            #loss = loss_use(output, mask)
            loss,edge_loss,bregular,sregular = loss_use(output,pred_edge, mask,edge,alpha1,alpha2)
            if torch.is_tensor(output):
                output = torch.sigmoid(output).data.cpu().numpy()
            if torch.is_tensor(mask):
                mask = mask.data.cpu().numpy()
                
            # iou = iou_score(output, mask)
            iou = calIOU(output, mask)
            dice = calDICE(output, mask)
            voe = calVOE(output, mask)
            rvd = calRVD(output, mask)
            sen = calSEN(output, mask)
            precision = calPrecision(output, mask)
            recall = calRecall(output, mask)

            log['loss'].update(loss.item(), args.batch_size)  # 对所有batch_size的损失求平均
            log['iou'].update(iou, args.batch_size)
            log['dice'].update(dice, args.batch_size)
            log['voe'].update(voe, args.batch_size)
            log['rvd'].update(rvd, args.batch_size)
            log['sen'].update(sen, args.batch_size)
            log['precision'].update(precision, args.batch_size)
            log['recall'].update(recall, args.batch_size)
            result = OrderedDict([('loss', log['loss'].avg),
                                  ('iou', log['iou'].avg),
                                  ('dice', log['dice'].avg),
                                 ('voe', log['voe'].avg),
                                 ('rvd', log['rvd'].avg),
                                  ('sen', log['sen'].avg),
                                 ('precision', log['precision'].avg),
                                 ('recall', log['recall'].avg)])
            # result = OrderedDict([('loss', log['loss'].avg), ('iou', log['iou'].avg)])

    return result

def retrain(data_generator, model ,epoch ,opt,alpha1,alpha2):
    result = [i for i in range(args.epochs)]
    log = {'loss': AverageMeter(),
            'edge_loss':AverageMeter(),
            'bregular':AverageMeter(),
            'sregular':AverageMeter(),
           'iou': AverageMeter(),
           'dice': AverageMeter(),
           'voe': AverageMeter(),
           'rvd': AverageMeter(),
           'sen':AverageMeter(),
           'precision': AverageMeter(),
           'recall': AverageMeter()}  # calculate average result of all samples
    # model.load_state_dict(torch.load(config['model_path'] + 'model_%s.pth'%args.block))
    model.train()

    for image, mask,edge in data_generator:
        image = image.to(device)
        mask = mask.to(device)
        edge = edge.to(device)

        loss_use = losses.TotalLoss().to(device)
        output,pred_edge = model(image)
        loss,edge_loss,bregular,sregular = loss_use(output,pred_edge, mask,edge,alpha1,alpha2)
        iou = calIOU(output, mask)
        dice = calDICE(output, mask)
        voe = calVOE(output, mask)
        rvd = calRVD(output, mask)
        sen = calSEN(output, mask)
        precision = calPrecision(output, mask)
        recall = calRecall(output, mask)

        opt.zero_grad()
        loss.backward()
        opt.step()

        log['loss'].update(loss.item(), args.batch_size)  # 对所有batch_size的损失求平均
        log['edge_loss'].update(edge_loss.item(),args.batch_size)
        log['bregular'].update(bregular.item(),args.batch_size)
        log['sregular'].update(sregular.item(),args.batch_size)
        log['iou'].update(iou, args.batch_size)
        log['dice'].update(dice, args.batch_size)
        log['voe'].update(voe, args.batch_size)
        log['rvd'].update(rvd, args.batch_size)
        log['sen'].update(sen, args.batch_size)
        log['precision'].update(precision, args.batch_size)
        log['recall'].update(recall, args.batch_size)

        result[epoch] = OrderedDict([('loss', log['loss'].avg),
                              ('edge_loss', log['edge_loss'].avg),
                              ('bregular', log['bregular'].avg),
                              ('sregular', log['sregular'].avg),
                              ('iou', log['iou'].avg),
                              ('dice', log['dice'].avg),
                              ('voe', log['voe'].avg),
                              ('rvd', log['rvd'].avg),
                              ('sen', log['sen'].avg),
                              ('precision', log['precision'].avg),
                              ('recall', log['recall'].avg)])
    return result




def train(data_generator, model, epoch, opt, alpha1,alpha2):

    result = [i for i in range(args.epochs)]
    log = {'loss': AverageMeter(),
            'edge_loss':AverageMeter(),
            'bregular':AverageMeter(),
            'sregular':AverageMeter(),
           'iou': AverageMeter(),
           'dice': AverageMeter(),
           'voe': AverageMeter(),
           'rvd': AverageMeter(),
           'sen':AverageMeter(),
           'precision': AverageMeter(),
           'recall': AverageMeter()}
    # log = {'loss': AverageMeter(), 'iou': AverageMeter()}  # calculate average result of all samples
    model.train()

    for image, mask,edge in data_generator:
        image = image.to(device)
        mask = mask.to(device)
        edge = edge.to(device)
        
        # print(mask.size())

        loss_use = losses.TotalLoss().to(device)
        output,pred_edge = model(image)
        # print(output.size())
        # print(pred_edge.size())
        loss,edge_loss,bregular,sregular = loss_use(output,pred_edge, mask,edge,alpha1,alpha2)
        opt.zero_grad()
        loss.backward()
        opt.step()


        if torch.is_tensor(output):
            output = torch.sigmoid(output).data.cpu().numpy()
        if torch.is_tensor(mask):
            mask = mask.data.cpu().numpy()
        # iou = iou_score(output, mask)
        iou = calIOU(output, mask)
        dice = calDICE(output, mask)
        voe = calVOE(output, mask)
        rvd = calRVD(output, mask)
        sen = calSEN(output, mask)
        precision = calPrecision(output, mask)
        recall = calRecall(output, mask)

       

        log['loss'].update(loss.item(), args.batch_size)  # 对所有batch_size的损失求平均
        log['edge_loss'].update(edge_loss.item(),args.batch_size)
        log['bregular'].update(bregular.item(),args.batch_size)
        log['sregular'].update(sregular.item(),args.batch_size)
        log['iou'].update(iou, args.batch_size)
        log['dice'].update(dice, args.batch_size)
        log['voe'].update(voe, args.batch_size)
        log['rvd'].update(rvd, args.batch_size)
        log['sen'].update(sen, args.batch_size)
        log['precision'].update(precision, args.batch_size)
        log['recall'].update(recall, args.batch_size)

        result[epoch] = OrderedDict([('loss', log['loss'].avg),
                              ('edge_loss', log['edge_loss'].avg),
                              ('bregular', log['bregular'].avg),
                              ('sregular', log['sregular'].avg),
                              ('iou', log['iou'].avg),
                              ('dice', log['dice'].avg),
                              ('voe', log['voe'].avg),
                              ('rvd', log['rvd'].avg),
                              ('sen', log['sen'].avg),
                              ('precision', log['precision'].avg),
                              ('recall', log['recall'].avg)])
    return result


def main():
    config = all_config(args.mode, args.fuc, args.model)
    os.makedirs(config['model_path'],exist_ok=True)
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers,
              'drop_last': True}

    if args.train_mode == 'train':
        cudnn.benchmark = True

        if args.model == 'SEUNet':
            model = SEUNet(**config)
        elif args.model == 'UNet':
            model = UNet3Plus()
            # model = UNet(**config)
        elif args.model == 'Transformer':
            # args.block = 'VesselTBlock2'
            #model = VTTransNet(CONFIGS['R50-ViT-L_16'],512,block=args.block)
            model=TransSimUNet(CONFIGS['R50-ViT-B_16'])


        model = model.to(device)

        total_data = [os.path.basename(i) for i in glob(config['image_path'] + "*.png")]
        train_data, val_data = train_test_split(total_data, test_size=0.2, random_state=41)

        train_set = EdgeDataset(train_data, 'train', **config)
        val_set = EdgeDataset(val_data, 'val', **config)

        train_generator = data.DataLoader(train_set, **params)
        val_generator = data.DataLoader(val_set, **params)

        max_iou = 0
        alpha1 = 0.
        alpha2 = 0.
        #opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)

        print("------------- Begin " + args.fuc + " model training -------------")
        for epoch in range(args.epochs):
            if epoch > args.decay_epoch:
                alpha1 = 0.3
                alpha2 = 0.5

            result_train = train(train_generator, model, epoch, opt, alpha1,alpha2)     # 开始训练模型
            # print('Training at Epoch [%d/%d]' % (epoch, args.epochs)    # 在每个epoch上，模型在训练集上的表现
            #       + ', loss: ' + str(result_train[epoch]['loss']), ' iou: ' + str(result_train[epoch]['iou']))
            print('Training at Epoch [%d/%d]' % (epoch, args.epochs)    # 在每个epoch上，模型在训练集上的表现
                  + ', loss: ' + str(result_train[epoch]['loss']),
                  ' edge_loss: ' + str(result_train[epoch]['edge_loss']),
                  ' bregular: ' + str(result_train[epoch]['bregular']),
                  ' sregular: ' + str(result_train[epoch]['sregular']),
                  ' iou: ' + str(result_train[epoch]['iou']),
                  ' dice: ' + str(result_train[epoch]['dice']),
                  ' voe: ' + str(result_train[epoch]['voe']),
                  ' rvd: ' + str(result_train[epoch]['rvd']),
                  ' sen: ' + str(result_train[epoch]['sen']),
                  ' precision: ' + str(result_train[epoch]['precision']),
                  ' recall: ' + str(result_train[epoch]['recall'])
                  )

            result_val = test(val_generator, model,alpha1,alpha2)        # 在每个epoch上，模型在验证集上的表现
            # print('Validation, loss: ' + str(result_val['loss']) + ' iou: ' + str(result_val['iou']))
            print('Validation, loss: ' + str(result_val['loss'])
                  + ' iou: ' + str(result_val['iou'])
                  + ' dice: ' + str(result_val['dice'])
                  + ' voe: ' + str(result_val['voe'])
                  + ' rvd: ' + str(result_val['rvd'])
                  + ' sen: ' + str(result_val['sen'])
                  + ' precision: ' + str(result_val['precision'])
                  + ' recall: ' + str(result_val['recall'])
                  )

            scheduler.step()

            if result_val['iou'] > max_iou:         # 如果模型在该验证集上表现最好，则保存此时的模型
                torch.save(model, config['model_path'] + 'model_%s.pth'%args.block)
                print("=======> saved best model <=======")
                max_iou = result_val['iou']

            torch.cuda.empty_cache()

    elif args.train_mode == 'test':
        print("------------- Begin " + args.fuc + " model testing -------------")

        test_data = [os.path.basename(i) for i in glob(config['image_path'] + "*.png")]
        test_set = Dataset(test_data, 'test', **config)

        test_generator = data.DataLoader(test_set, **params)


        model = torch.load(config['model_path'] + 'model.pth')
        result_test = test(test_generator, model)
        print('Test result, loss: ' + str(result_test['loss']) + ' iou: ' + str(result_test['iou']))


    else:
        
        cudnn.benchmark = True

        if args.model == 'SEUNet':
            model = SEUNet(**config)
        elif args.model == 'UNet':
            model = UNet3Plus()
            # model = UNet(**config)
        elif args.model == 'Transformer':
            model=TransSimUNet(CONFIGS['R50-ViT-B_16'])

        # model = model.to(device)
        model = torch.load('save_model/pretrain_vessel/model_%s.pth'%args.block,map_location=device)
        model.to(device)
        model.train()

        total_data = [os.path.basename(i) for i in glob(config['image_path'] + "*.png")]
        train_data, val_data = train_test_split(total_data, test_size=0.2, random_state=41)

        train_set = EdgeDataset(train_data, 'train', **config)
        val_set = EdgeDataset(val_data, 'val', **config)

        train_generator = data.DataLoader(train_set, **params)
        val_generator = data.DataLoader(val_set, **params)

        max_iou = 0
        alpha1 = 0.3
        alpha2 = 0.5
        #opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)

        print("------------- Begin " + args.fuc + " model retraining -------------")
        for epoch in range(args.epochs):
            if epoch > args.decay_epoch:
                alpha1 = 0.3
                alpha2 = 0.3
            result_train = train(train_generator, model, epoch, opt,alpha1,alpha2)     # 开始训练模型
            print('Training at Epoch [%d/%d]' % (epoch, args.epochs)    # 在每个epoch上，模型在训练集上的表现
                  + ', loss: ' + str(result_train[epoch]['loss']),
                  ' edge_loss: ' + str(result_train[epoch]['edge_loss']),
                  ' bregular: ' + str(result_train[epoch]['bregular']),
                  ' sregular: ' + str(result_train[epoch]['sregular']),
                  ' iou: ' + str(result_train[epoch]['iou']),
                  ' dice: ' + str(result_train[epoch]['dice']),
                  ' voe: ' + str(result_train[epoch]['voe']),
                  ' rvd: ' + str(result_train[epoch]['rvd']),
                  ' sen: ' + str(result_train[epoch]['sen']),
                  ' precision: ' + str(result_train[epoch]['precision']),
                  ' recall: ' + str(result_train[epoch]['recall'])
                  )
            result_val = test(val_generator, model)        # 在每个epoch上，模型在验证集上的表现
            print('Validation, loss: ' + str(result_val['loss'])
                  + ' iou: ' + str(result_val['iou'])
                  + ' dice: ' + str(result_val['dice'])
                  + ' voe: ' + str(result_val['voe'])
                  + ' rvd: ' + str(result_val['rvd'])
                  + ' sen: ' + str(result_val['sen'])
                  + ' precision: ' + str(result_val['precision'])
                  + ' recall: ' + str(result_val['recall'])
                  )
            scheduler.step()

            if result_val['iou'] > max_iou:         # 如果模型在该验证集上表现最好，则保存此时的模型
                torch.save(model, config['model_path'] + 'model_%s.pth'%args.block)
                print("=======> saved best model <=======")
                max_iou = result_val['iou']

            torch.cuda.empty_cache()


        

if __name__ == '__main__':
    main()
