import os
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn import metrics
from pytorch_msssim import ms_ssim, ssim
from mvtec_exp.p33.git_msssim import *
from mvtec_exp.p33.mvtec_module import *
from mvtec_exp.p33.gsvdd_module import gsvddGenerator
from mvtec_exp.p33.mvtex_data_loader import MvtecDataLoader
import Helper

torch.cuda.empty_cache()

device = torch.device("cuda:0")
print(">> Device Info: {} is in use".format(torch.cuda.get_device_name(0)))

saver_count = 0

BATCH_SIZE = 1
############################ Parameters ############################
latent_dimension = 512


category = {
            # 1: "bottle",
            # 2: "hazelnut",
            # 3: "capsule",
            # 4: "metal_nut",
            # 5: "leather",
            6: "pill",
            # 7: "wood",
            # 8: "carpet",
            # 9: "tile",
            # 10: "grid",
            # 11: "cable",
            # 12: "transistor",
            # 13: "toothbrush",
            # 14: "screw",
            # 15: "zipper"
            }


# ####################################################################

def load_train(train_path):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        # torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    imagenet_data = MvtecDataLoader(train_path, transform=transform)
    # imagenet_data = torchvision.datasets.ImageFolder(test_path, transform=transform)

    train_data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=6,
                                                    pin_memory=True)
    return train_data_loader


def load_test(test_path):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        # torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    imagenet_data = MvtecDataLoader(test_path, transform=transform)
    # imagenet_data = torchvision.datasets.ImageFolder(test_path, transform=transform)

    valid_data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                    batch_size=BATCH_SIZE,
                                                    num_workers=6,
                                                    pin_memory=True)
    return valid_data_loader


AUC_ALL = []
# require_auc_plot = True
# if require_auc_plot:
#     # auc_plt = plt.figure()
#     # plt.title('MS-SSIM Baseline')#.format(mem_size, mem_lamb, NORMAL_NUM_LIST))
#     plt.plot([0, 1], [0, 1], linestyle='--')
#     plt.xlim([-0.005, 1.005])
#     plt.ylim([-0.005, 1.005])
#     plt.xlabel('FPR')
#     plt.ylabel('TPR')

all_group = None


def extract_patch(data_tmp):
    tmp = None
    for i in range(8):
        for j in range(8):
            tmp = data_tmp[:, :, i, j, :, :] if i == 0 and j == 0 \
                else torch.cat((tmp, data_tmp[:, :, i, j, :, :]), dim=0)
    return tmp


for key in category:
    NORMAL_NUM = category[key]
    print('Current Item: {}'.format(NORMAL_NUM))

    train_path = '/home/yuyuanliu/Desktop/Deep Learning/public_data/MVTec_AD/{}/train/'.format(NORMAL_NUM)
    test_root = '/home/yuyuanliu/Desktop/Deep Learning/public_data/MVTec_AD/{}/test/'.format(NORMAL_NUM)
    gt_root = '/home/yuyuanliu/Desktop/Deep Learning/public_data/MVTec_AD/{}/ground_truth/'.format(NORMAL_NUM)

    # recorder = Recorder(Experiment_name, 'CIFAR-10_No.{}'.format(str(NORMAL_NUM)))

    # generator     = twoin1Generator256(64, latent_dimension=latent_dimension)
    # discriminator = VisualDiscriminator256(64)
    # gsvdd_gen = gsvddGenerator(64, latent_dimension=128)
    generator = torch.load('./check_points/No.{}_p32_NMSL_wd_sig_f_2.5_rec_f_10_svd_f_0.1/No.{}_g'.format(NORMAL_NUM, NORMAL_NUM))
    # discriminator.load_state_dict(torch.load('check_points/No.{}_gsvdd/No.{}_d_9999'.format(NORMAL_NUM, NORMAL_NUM)))
    gsvdd_gen = torch.load('/media/yuyuanliu/T7 Touch/2020-08-31/gsvdd_inter/check_points/No.{}_gsvdd_msssim_ft/No.{}_g'.format(NORMAL_NUM, NORMAL_NUM))

    criterion = torch.nn.MSELoss()
    l1_criterion = torch.nn.L1Loss()
    # ent_criterion = NegEntropyLoss()

    generator.to(device)
    gsvdd_gen.to(device)
    # discriminator.to(device)

    generator.eval()
    gsvdd_gen.eval()
    # discriminator.eval()

    y = []
    score = []
    score_recon = []
    score_gsvdd = []
    normal_mse_loss = []
    abnormal_mse_loss = []
    test_root = '/home/yuyuanliu/Desktop/Deep Learning/public_data/MVTec_AD/{}/test/'.format(NORMAL_NUM)
    list_test = os.listdir(test_root)
    # print("gsvdd_sigma: {}".format(generator.sigma))

    data_range = 2.1179 + 2.6400
    with torch.no_grad():
        for i in range(len(list_test)):
            current_defect = list_test[i]

            test_path = test_root + "{}".format(current_defect)
            valid_dataset_loader = load_test(test_path)
            for index, (images, label) in enumerate(valid_dataset_loader):
                # img = images.to(device)
                img_whole = images.to(device)
                img_tmp = img_whole.unfold(2, 32, 32).unfold(3, 32, 32)
                img = extract_patch(img_tmp)
                generate_result = generator(img)
                weight = 0.85
                ms_ssim_batch_wise = 1 - ms_ssim(img, generate_result, data_range=data_range,
                                                 size_average=True, win_size=3)
                # ms_ssim_batch_wise = 1 - ssim(img, generate_result, data_range=data_range,
                #                               size_average=True, win_size=11)
                l1_batch_wise = l1_criterion(img, generate_result) / data_range
                ms_ssim_l1 = weight * ms_ssim_batch_wise + (1 - weight) * l1_batch_wise

                gsvdd_result = gsvdd_gen.encoder(img_whole)
                diff = (gsvdd_result - gsvdd_gen.c) ** 2
                dist = -1 * torch.sum(diff, dim=1) / gsvdd_gen.sigma
                guass_svdd_loss = 1 - torch.exp(dist)


                score_recon = float(ms_ssim_l1.cpu().detach().numpy())
                score_gsvdd = float(guass_svdd_loss[0].cpu().detach().numpy())
                anormaly_score = score_recon + score_gsvdd


                # if current_defect == "good":
                #     grid_imgs = torchvision.utils.make_grid(img, normalize=True, nrow=8,
                #                                             range=(-2.1179, 2.6400))
                #     np_grid_imgs = grid_imgs.cpu().detach().numpy()
                #     temp = numpy.transpose(np_grid_imgs, (1, 2, 0)).squeeze().astype(numpy.float32)
                #     plt.imshow(temp)
                #     plt.show()
                #     p=1
                #
                #
                #     grid_imgs = torchvision.utils.make_grid(generate_result, normalize=True, nrow=8,
                #                                             range=(-2.1179, 2.6400))
                #     np_grid_imgs = grid_imgs.cpu().detach().numpy()
                #     temp = numpy.transpose(np_grid_imgs, (1, 2, 0)).squeeze().astype(numpy.float32)
                #     plt.imshow(temp)
                #     plt.show()
                #     p=1
                score.append(anormaly_score)
                if label[0] == "good":
                    # normal_gsvdd.append(float(guass_svdd_loss.cpu().detach().numpy()))
                    normal_mse_loss.append(anormaly_score)
                    y.append(0)
                else:
                    # abnormal_gsvdd.append(float(guass_svdd_loss.cpu().detach().numpy()))
                    abnormal_mse_loss.append(anormaly_score)
                    y.append(1)


    ###################################################
    Helper.plot_2d_chart(x1=numpy.arange(0, len(normal_mse_loss)), y1=normal_mse_loss, label1='normal_loss',
                         x2=numpy.arange(len(normal_mse_loss), len(normal_mse_loss) + len(abnormal_mse_loss)),
                         y2=abnormal_mse_loss, label2='abnormal_loss',
                         title=NORMAL_NUM)

    fpr, tpr, thresholds = metrics.roc_curve(y, score, pos_label=1)
    auc_result = auc(fpr, tpr)
    print(auc_result)
    AUC_ALL.append(auc_result)

# average_auc = sum(AUC_ALL) / len(AUC_ALL)
# print('Average AUC:', average_auc)
# plt.title('Average AUC: {}'.format(average_auc))
# plt.grid()
# plt.show()


# plt.savefig("check_points/auc_all.png")
# plt.show()

