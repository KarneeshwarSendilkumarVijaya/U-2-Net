import argparse
import heapq
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import glob
import os
from service.data_loader import RescaleT, Rescale
from service.data_loader import RandomCrop
from service.data_loader import ToTensorLab
from service.data_loader import SalObjDataset
from utils.logger_details import get_logger

from model import U2NET
from model import U2NETP
from utils.gcs_io import get_training_files_from_gcs, save_model_to_gcs, download_blob_from_gcs
from commons.constants import THRESHOLD, MODEL_SAVE_COUNT, MODEL_CONVERGENCE_COUNT
from commons.constants import PRETRAINED_MODEL_FOLDER, BEST_MODEL_FOLDER, GCS_BUCKET, LOG_PADDING_WIDTH, LOG_PADDING_SYMBOL

log = get_logger('__main__')

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
best_10_models = []


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    log.info("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f" % (
        loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
        loss5.data.item(),
        loss6.data.item()))

    return loss0, loss


def model_heap_check(current_model, current_loss):
    current = (-1*current_loss, current_model)

    if 0 <= len(best_10_models) < MODEL_SAVE_COUNT:
        heapq.heappush(best_10_models, current)
        return current_model, None

    while len(best_10_models) > 10:
        heapq.heappop(best_10_models)

    max = heapq.heappop(best_10_models)
    max_loss = -1*max[0]
    max_model = max[1]

    if current_loss < max_loss:
        heapq.heappush(best_10_models, current)
        return current_model, max_model

    heapq.heappush(best_10_models, max)
    return None, None


# ------- 2. set the directory of training dataset --------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100000, type=int)
    parser.add_argument("--save_frq", default=2000, type=int)
    parser.add_argument("--datasets", default="xsmall")
    parser.add_argument("--model_name", default="u2net")
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--existing_model", default='', required=False) # pretrained or best
    parser.add_argument("--best_model_name", default='', required=False)
    # parser.add_argument("--load_from_gcs", default=False)
    args = parser.parse_args()

    log.info(f"Epoch = {args.epochs}".center(LOG_PADDING_WIDTH, LOG_PADDING_SYMBOL))
    log.info(f"Save Frequency = {args.save_frq}".center(LOG_PADDING_WIDTH, LOG_PADDING_SYMBOL))
    log.info(f"Dataset = {args.datasets}".center(LOG_PADDING_WIDTH, LOG_PADDING_SYMBOL))
    log.info(f"Model Name = {args.model_name}".center(LOG_PADDING_WIDTH, LOG_PADDING_SYMBOL))
    log.info(f"Batch Size = {args.batch_size}".center(LOG_PADDING_WIDTH, LOG_PADDING_SYMBOL))
    log.info(f"Existing Model = {args.existing_model}".center(LOG_PADDING_WIDTH, LOG_PADDING_SYMBOL))
    log.info(f"Best Model Name = {args.best_model_name}".center(LOG_PADDING_WIDTH, LOG_PADDING_SYMBOL))

    existing_model = args.existing_model
    best_model_name = args.best_model_name
    model_name = args.model_name  # 'u2net' or 'u2netp'

    # data_dir = os.path.join(os.getcwd(), '../train_data' + os.sep)
    # tra_image_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'im_aug' + os.sep)
    # tra_label_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'gt_aug' + os.sep)

    image_ext = '.jpg'
    label_ext = '.png'

    # model_dir = os.path.join(os.getcwd(), '../saved_models', model_name + os.sep)

    epoch_num = args.epochs
    dataset = args.datasets
    batch_size_train = args.batch_size
    batch_size_val = 1
    train_num = 0
    val_num = 0

    # tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
    # tra_lbl_name_list = glob.glob(data_dir + tra_label_dir + '*' + label_ext)
    #
    tra_img_name_list, tra_lbl_name_list = get_training_files_from_gcs(dataset)

    log.info("---")
    log.info(f"train images: {len(tra_img_name_list)}".center(LOG_PADDING_WIDTH, LOG_PADDING_SYMBOL))
    log.info(f"train labels: {len(tra_lbl_name_list)}".center(LOG_PADDING_WIDTH, LOG_PADDING_SYMBOL))
    log.info("---")

    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            # RandomCrop(288), # no need to crop
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

    # ------- 3. define model --------
    # define the net
    if model_name == 'u2net':
        net = U2NET(3, 1)
    elif model_name == 'u2netp':
        net = U2NETP(3, 1)

    if existing_model == 'pretrained':
        log.info("Loading pretrained model".center(LOG_PADDING_WIDTH, LOG_PADDING_SYMBOL))
        download_blob_from_gcs('pretrained_model.pth', f'{PRETRAINED_MODEL_FOLDER}/u2net.pth')
        net.load_state_dict(torch.load('pretrained_model.pth'))
    elif existing_model == 'best':
        log.info("Loading best model".center(LOG_PADDING_WIDTH, LOG_PADDING_SYMBOL))
        download_blob_from_gcs('best_model.pth', f'{BEST_MODEL_FOLDER}/{best_model_name}')
        net.load_state_dict(torch.load('best_model.pth'))

    if torch.cuda.is_available():
        net.cuda()

    # ------- 4. define optimizer --------
    log.info("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    log.info("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = args.save_frq  # save the model every 2000 iterations
    last_10_loss = []
    convergence = False

    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            log.info("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f \n" % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
                running_tar_loss / ite_num4val))

            if ite_num % save_frq == 0:
                model_to_save, model_to_delete = model_heap_check(model_name + "_bce_itr_%d_train_%3f_tar_%3f.pth" % (
                    ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val), running_tar_loss / ite_num4val)
                log.info(f"saving model = {model_to_save}".center(LOG_PADDING_WIDTH, LOG_PADDING_SYMBOL))
                log.info(f"deleting model = {model_to_delete}".center(LOG_PADDING_WIDTH, LOG_PADDING_SYMBOL))
                if model_to_save is not None:
                    save_model_to_gcs(net, model_to_save, model_to_delete)
                # torch.save(net.state_dict(), model_dir + model_name + "_bce_itr_%d_train_%3f_tar_%3f.pth" % (
                #     ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

                last_10_loss.append(running_tar_loss / ite_num4val)
                if len(last_10_loss) > MODEL_CONVERGENCE_COUNT:
                    last_10_loss.pop(0)
                if len(last_10_loss) == MODEL_CONVERGENCE_COUNT:
                    if all(abs(running_tar_loss / ite_num4val - last_10_loss[j]) < THRESHOLD for j in range(1, len(last_10_loss))):
                        convergence = True
                        break

                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0

        if convergence:
            break
