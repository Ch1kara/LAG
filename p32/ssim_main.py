import os
import numpy
from Recorder import Recorder
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import autograd
from torch import optim
import Helper
import torch.nn.init as init
from timeit import default_timer as timer
import cv2
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from pytorch_msssim import ms_ssim, ssim
from tqdm import tqdm
import numpy as np
# from mvtec_exp.p33.git_msssim import *
from p32.ssim_module import *
from torch.autograd import Variable
from p32.lag_data_loader import *
# torch.autograd.set_detect_anomaly(True)
import torchvision.transforms.functional as TF

device = torch.device("cuda:1")
print(">> Device Info: {} is in use".format(device))

DIM = 32                    # Model dimensionality
CRITIC_ITERS = 5            # How many iterations to train the critic for
GENER_ITERS = 1
N_GPUS = 1                  # Number of GPUs
BATCH_SIZE = 1            # Batch size. Must be a multiple of N_GPUS
global END_ITER
MAX_EPOCH = 256
# START_ITER = 0
# END_ITER = int((5000/BATCH_SIZE) * 256)            # How many iterations to train for
# END_ITER = int((300/BATCH_SIZE) * 200)
# print(END_ITER)
LAMBDA = 10                 # Gradient pena1lty lambda hyperparameter
OUTPUT_DIM = 32 * 32 * 3    # Number of pixels in each image
BEST_AUC = 0
NORMAL_NUM = 'LAG'
############################ Parameters ############################
latent_dimension = 128

category = {
            1: "train_set",
            }

data_range = 2.1179 + 2.6400
ssim_weights = [0.0516, 0.3295, 0.3463, 0.2726]
# data_range = 2

# generator.sigma = init_sigma(train_dataset_loader, generator) * sig_f
# encoder_loss = rec_f * ms_ssim_l1 + 0.1 * reg_inter + svd_f * svdd_loss
sig_f = 2
rec_f = 10
svd_f = 0.1

####################################################################

USE_SSIM = True

LR = 1e-4       # 0.0001
one = torch.tensor(1, dtype=torch.float)
mone = one * -1
one = one.to(device)
mone = mone.to(device)
# sigma = torch.tensor(1, dtype=torch.float).to(device)



nu = torch.tensor(0.1, dtype=torch.float).to(device)

mean_dist = None
# objective = 'soft-boundary'
# objective = 'HARD'


mse_criterion = torch.nn.MSELoss()
l1_criterion = torch.nn.L1Loss()
bce_criterion = torch.nn.BCELoss()

sigbce_criterion = torch.nn.BCEWithLogitsLoss()


def weights_init(m):
    if isinstance(m, MyConvo2d):
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, torch.nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]


def load_train(train_path):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.RandomChoice([
            torchvision.transforms.RandomApply([
                torchvision.transforms.RandomRotation(degrees=(90, 90)),
            ], p=0.5),
            # torchvision.transforms.RandomApply([
            #     torchvision.transforms.RandomRotation(degrees=(180, 180)),
            # ], p=0.5),
            torchvision.transforms.RandomApply([
                torchvision.transforms.RandomRotation(degrees=(-90, -90)),
            ], p=0.5),
        ]),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    imagenet_data = torchvision.datasets.ImageFolder(train_path, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=5,
                                                    pin_memory=True)
    return train_data_loader, imagenet_data.__len__()


def load_test(test_path):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    imagenet_data = LAGDataLoader(test_path, transform=transform)
    # imagenet_data = torchvision.datasets.ImageFolder(test_path, transform=transform)

    valid_data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=5,
                                                    pin_memory=True)
    return valid_data_loader


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=0, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """
    if max_iter == 0:
        raise Exception("买菜必涨价 超级加倍")
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    lr = init_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def extract_patch(data_tmp):
    tmp = None
    _,_,a,b,_,_ = data_tmp.shape
    for i in range(a):
        for j in range(b):
            tmp = data_tmp[:, :, i, j, :, :] if i == 0 and j == 0 \
                else torch.cat((tmp, data_tmp[:, :, i, j, :, :]), dim=0)
    return tmp


def init_c(DataLoader, net, eps=0.1):
    c = torch.zeros((1, latent_dimension)).to(device)
    net.eval()
    n_samples = 0
    print("Estimating Center ...")
    with torch.no_grad():
        for index, (images, label) in enumerate(tqdm(DataLoader, position=0)):
            img_org = images.to(device)
            img_tmp = img_org.unfold(2, 32, 32).unfold(3, 32, 32)
            # img = extract_patch(img_tmp)
            img = img_tmp.contiguous().view(-1, 3, 32, 32)
            outputs = net.encoder(img)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c


def init_sigma(DataLoader, net):
    generator.sigma = None
    net.eval()
    tmp_sigma = torch.tensor(0.0, dtype=torch.float).to(device)
    n_samples = 0
    print("Estimating Standard Deviation ...")
    with torch.no_grad():
        for index, (images, label) in enumerate(tqdm(DataLoader, position=0)):
            img_org = images.to(device)
            img_tmp = img_org.unfold(2, 32, 32).unfold(3, 32, 32)
            img = img_tmp.contiguous().view(-1, 3, 32, 32)
            latent_z = net.encoder(img)
            diff = (latent_z - generator.c) ** 2
            tmp = torch.sum(diff.detach(), dim=1)
            if (tmp.mean().detach() / sig_f) < 1:
                tmp_sigma += 1
            else:
                tmp_sigma += tmp.mean().detach() / sig_f
            n_samples += 1
    tmp_sigma /= n_samples
    return tmp_sigma


def train(NORMAL_NUM,
          generator, discriminator,
          optimizer_g, optimizer_d):
    global test_auc
    train_path = '/home/yuanhong/Documents/Public_Dataset/LAG_AD/Train'
    START_ITER = 0
    train_size = len(os.listdir(train_path))
    generator.train()
    train_dataset_loader, train_size = load_train(train_path)
    END_ITER = int((train_size / BATCH_SIZE) * MAX_EPOCH)

    generator.c = None
    generator.sigma = None
    # generator.c = torch.rand((1, 128)).to(device)
    # generator.sigma = torch.tensor(10).to(device)

    generator.c = init_c(train_dataset_loader, generator)
    generator.sigma = init_sigma(train_dataset_loader, generator)



    print("gsvdd_sigma: {}".format(generator.sigma))

    train_data = iter(train_dataset_loader)
    process = tqdm(range(START_ITER, END_ITER), desc='{AUC: }')

    for iteration in process:
        poly_lr_scheduler(optimizer_d, init_lr=LR, iter=iteration, max_iter=END_ITER)
        poly_lr_scheduler(optimizer_g, init_lr=LR, iter=iteration, max_iter=END_ITER)

        # --------------------- Loader ------------------------
        batch = next(train_data, None)
        if batch is None:
            train_dataset_loader, _ = load_train(train_path)
            train_data = iter(train_dataset_loader)
            batch = train_data.next()
        batch = batch[0]  # batch[1] contains labels
        batch_data = batch.to(device)
        data_tmp = batch_data.unfold(2, 32, 32).unfold(3, 32, 32)
        real_data = extract_patch(data_tmp)

        # --------------------- TRAIN E ------------------------
        optimizer_g.zero_grad()
        latent_z = generator.encoder(real_data)
        fake_data = generator(real_data)

        # Reconstruction loss
        weight = 0.85
        ms_ssim_batch_wise = 1 - ms_ssim(real_data, fake_data, data_range=data_range,
                                         size_average=True, win_size=3, weights=ssim_weights)
        # ms_ssim_batch_wise = 1 - ssim(real_data, fake_data, data_range=data_range,
        #                               size_average=True, win_size=11)
        l1_batch_wise = l1_criterion(real_data, fake_data)/data_range
        ms_ssim_l1 = weight * ms_ssim_batch_wise + (1 - weight) * l1_batch_wise

        ############ Interplote ############
        e1 = torch.flip(latent_z, dims=[0])
        alpha = torch.FloatTensor(BATCH_SIZE, 1).uniform_(0, 0.5).to(device)
        e2 = alpha * latent_z + (1 - alpha) * e1
        g2 = generator.generate(e2)
        reg_inter = torch.mean(discriminator(g2) ** 2)

        ############ DSVDD ############
        diff = (latent_z - generator.c) ** 2
        dist = -1 * (torch.sum(diff, dim=1) / generator.sigma)
        svdd_loss = torch.mean(1 - torch.exp(dist))

        encoder_loss = rec_f * ms_ssim_l1 + svd_f * svdd_loss + 0.1 * reg_inter
        encoder_loss.backward()
        optimizer_g.step()

        # ------------------- Train D -------------------
        optimizer_d.zero_grad()
        g2 = generator.generate(e2).detach()
        fake_data = generator(real_data).detach()
        d_loss_front = torch.mean((discriminator(g2) - alpha) ** 2)
        gamma = 0.2
        tmp = fake_data + gamma * (real_data - fake_data)
        d_loss_back = torch.mean(discriminator(tmp) ** 2)

        d_loss = d_loss_front + d_loss_back
        d_loss.backward()
        optimizer_d.step()

        if recorder is not None:
            recorder.record(loss=svdd_loss, epoch=int(iteration / BATCH_SIZE),
                            num_batches=len(train_data), n_batch=iteration, loss_name='DSVDD')

            recorder.record(loss=torch.mean(dist), epoch=int(iteration / BATCH_SIZE),
                            num_batches=len(train_data), n_batch=iteration, loss_name='DIST')

            recorder.record(loss=ms_ssim_batch_wise, epoch=int(iteration / BATCH_SIZE),
                            num_batches=len(train_data), n_batch=iteration, loss_name='MS-SSIM')

            recorder.record(loss=l1_batch_wise, epoch=int(iteration / BATCH_SIZE),
                            num_batches=len(train_data), n_batch=iteration, loss_name='L1')

        if iteration % int((train_size / BATCH_SIZE) * 5) == 0 or iteration == END_ITER - 1:
            # recorder.log_images(batch_data, 1, iteration / BATCH_SIZE, iteration, len(train_data),
            #                     title="normal", normalize=True, range=(-2.1179, 2.6400))
            is_end = True if iteration == END_ITER-1 else False
            test_auc = validation(NORMAL_NUM, iteration, generator, discriminator, real_data, fake_data, is_end)
            process.set_description("{AUC: %.5f}" % test_auc)

        # if iteration == END_ITER - 1:
            if iteration > (END_ITER - 1) * 0.5:
                if test_auc - BEST_AUC > 0.0001:
                    auc_folder_tmp = ckpt_path+'/epoch_{}_auc_{}'.format((iteration*BATCH_SIZE)/train_size, test_auc)
                    if not os.path.exists(auc_folder_tmp):
                        os.mkdir(path=auc_folder_tmp)
                    torch.save(generator, auc_folder_tmp + '/No.{}_g.pth'.format(str(NORMAL_NUM)))
                    torch.save(discriminator.state_dict(), auc_folder_tmp + '/No.{}_d'.format(str(NORMAL_NUM)))
                    BEST_AUC = test_auc

                # opt_path = ckpt_path + '/optimizer'
                # if not os.path.exists(opt_path):
                #     os.mkdir(path=opt_path)
                # torch.save(optimizer_g.state_dict(), ckpt_path + '/optimizer/g_opt.pth')
                # torch.save(optimizer_d.state_dict(), ckpt_path + '/optimizer/d_opt.pth')
                # torch.save(optimizer_e.state_dict(), ckpt_path + '/optimizer/e_opt_{}.pth'.format(str(iteration)))


def validation(NORMAL_NUM, iteration, generator, discriminator, real_data, fake_data, is_end):
    # discriminator.eval()
    generator.eval()
    # resnet.eval()
    y     = []
    score = []
    normal_gsvdd = []
    abnormal_gsvdd = []
    normal_recon = []
    abnormal_recon = []
    test_path = '/home/yuanhong/Documents/Public_Dataset/LAG_AD/Test'
    list_test = os.listdir(test_root)

    with torch.no_grad():
        valid_dataset_loader = load_test(test_path)
        for index, (images, label) in enumerate(tqdm(valid_dataset_loader, position=0)):
            # img = five_crop_ready(images)
            img_tmp = images.to(device)
            img_tmp = img_tmp.unfold(2, 32, 32).unfold(3, 32, 32)
            # img     = extract_patch(img_tmp)
            img = img_tmp.contiguous().view(-1, 3, 32, 32)
            latent_z = generator.encoder(img)
            generate_result = generator(img)

            # Print tensorboard
            # if index == 0:
            #     temp = img[:8]
            #     j = 8
            #     for i in range(0, 8-1, 8):
            #         temp = torch.cat((temp, generate_result[i:i + 8]), 0)
            #         if j < 8-1:
            #             temp = torch.cat((temp, img[j:j + 4]), 0)
            #         j += 4
            if index == 3:
                grid_imgs = torchvision.utils.make_grid(generate_result, normalize=True, nrow=8, range=(-2.1179, 2.6400))
                np_grid_imgs = grid_imgs.cpu().detach().numpy()
                temp = np.transpose(np_grid_imgs, (1, 2, 0)).squeeze().astype(np.float32)

                recorder.log_images(images, 64, iteration / BATCH_SIZE, iteration, len(valid_dataset_loader),
                                    title="org", normalize=True, range=(-2.1179, 2.6400))

                recorder.log_images(temp, 64, iteration/BATCH_SIZE, iteration, len(valid_dataset_loader),
                                    title="rec", normalize=False)

            ############################## Normal #####################

            weight = 0.85

            ms_ssim_batch_wise = 1 - ms_ssim(img, generate_result, data_range=data_range,
                                             size_average=True, win_size=3, weights=ssim_weights)
            # ms_ssim_batch_wise = 1 - ssim(img, generate_result, data_range=data_range,
            #                               size_average=True, win_size=11)
            l1_batch_wise = l1_criterion(img, generate_result) / data_range
            ms_ssim_l1 = weight * ms_ssim_batch_wise + (1 - weight) * l1_batch_wise

            diff = (latent_z - generator.c) ** 2
            dist = -1 * torch.sum(diff, dim=1) / generator.sigma
            guass_svdd_loss = torch.mean(1 - torch.exp(dist))

            anormaly_score = (0.9 * ms_ssim_l1 + 0.1 * guass_svdd_loss).cpu().detach().numpy()
            score.append(float(anormaly_score))
            if label == 0:
                if is_end:
                    normal_gsvdd.append(float(guass_svdd_loss.cpu().detach().numpy()))
                    normal_recon.append(float(ms_ssim_l1.cpu().detach().numpy()))
                y.append(0)
            else:
                if is_end:
                    abnormal_gsvdd.append(float(guass_svdd_loss.cpu().detach().numpy()))
                    abnormal_recon.append(float(ms_ssim_l1.cpu().detach().numpy()))
                y.append(1)

        ###################################################
    if is_end:
        Helper.plot_2d_chart(x1=numpy.arange(0, len(normal_gsvdd)), y1=normal_gsvdd, label1='normal_loss',
                             x2=numpy.arange(len(normal_gsvdd), len(normal_gsvdd) + len(abnormal_gsvdd)),
                             y2=abnormal_gsvdd, label2='abnormal_loss',
                             title="{}: {}".format(NORMAL_NUM, "gsvdd loss"),
                             save_path="./plot/{}_gsvdd".format(NORMAL_NUM))
    # if True:
        Helper.plot_2d_chart(x1=numpy.arange(0, len(normal_recon)), y1=normal_recon, label1='normal_loss',
                             x2=numpy.arange(len(normal_recon), len(normal_recon) + len(abnormal_recon)),
                             y2=abnormal_recon, label2='abnormals_loss',
                             title="{}: {}".format(NORMAL_NUM, "recon loss"),
                             save_path="./plot/{}_gsvdd_{}".format(NORMAL_NUM, iteration))

    fpr, tpr, thresholds = metrics.roc_curve(y, score, pos_label=1)
    auc_result = auc(fpr, tpr)
    # tqdm.write(str(auc_result), end='.....................')
    auc_file = open(ckpt_path + "/auc.txt", "a")
    auc_file.write('Iter {}:            {}\r\n'.format(str(iteration), str(auc_result)))
    auc_file.close()
    return auc_result

BEST_AUC = 0
# NORMAL_NUM = category[key] if key.isdigit() else key

print('Current Item: {}'.format(NORMAL_NUM))

# train_path = '/home/yuanhong/Documents/Public_Dataset/LAG_AD/train_set'
test_root = '/home/yuanhong/Documents/Public_Dataset/LAG_AD/test_set'
gt_root = '/home/yuanhong/Documents/Public_Dataset/LAG_AD/test_set/attention_map'

# current_ckpt = "23999"
Experiment_name = 'No.{}_p32'.format(str(NORMAL_NUM))
# Experiment_name = 'No.{}_vae'.format(str(NORMAL_NUM))

recorder = Recorder(Experiment_name, '{}_AD'.format(str(NORMAL_NUM)))

if not os.path.exists('plot'):
    os.mkdir(path='plot')

if not os.path.exists('check_points'):
    os.mkdir(path='check_points')
ckpt_path = 'check_points/{}'.format(Experiment_name)

if not os.path.exists(ckpt_path):
    os.mkdir(path=ckpt_path)
auc_file = open(ckpt_path + "/auc.txt", "w")
auc_file.close()

generator     = twoin1Generator(64, latent_dimension=latent_dimension)
discriminator = VisualDiscriminator(64)

path = './Encoder_lr_scheduler_ckpt'
generator.pretrain.load_state_dict(torch.load(path))
for param in generator.pretrain.parameters():
    param.requires_grad = False

generator = generator.to(device)
discriminator = discriminator.to(device)

optimizer_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0, 0.9), weight_decay=1e-6)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0, 0.9))

train(NORMAL_NUM,
      generator,
      discriminator,
      optimizer_g,
      optimizer_d)
print(BEST_AUC)