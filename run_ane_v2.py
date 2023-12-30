import os
import piq
import torch
from piq import *
import numpy as np
import random
from itertools import chain

import PIL.Image as Image
from prettytable import PrettyTable

from torchvision.transforms.functional import to_tensor

import os
import os.path as osp
import itertools
import time


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def quality_metric(pred, gt, metric):
    loss_list = []
    assert len(pred) != 0 and len(pred) == len(gt)
    for pred_path, gt_path in zip(pred, gt):
        # img_pred = Image.open(pred_path).convert('L')
        # img_gt = Image.open(gt_path).convert('L')
        img_pred = Image.open(pred_path)
        img_gt = Image.open(gt_path)

        img_pred = to_tensor(img_pred).unsqueeze(0)
        img_gt = to_tensor(img_gt).unsqueeze(0)

        loss = metric(img_pred, img_gt)

        loss_list.append(loss.item())

    return sum(loss_list) / len(loss_list), loss_list


metric_no_ref = ['total_variation', 'brisque', 'inception_score']


def quality_metric_no_ref(pred, gt, metric):
    loss_list_a = []
    loss_list_b = []
    assert len(pred) != 0 and len(pred) == len(gt)
    for pred_path, gt_path in zip(pred, gt):
        # img_pred = Image.open(pred_path).convert('L')
        # img_gt = Image.open(gt_path).convert('L')
        img_pred = Image.open(pred_path)
        img_gt = Image.open(gt_path)

        img_pred = to_tensor(img_pred).unsqueeze(0)
        img_gt = to_tensor(img_gt).unsqueeze(0)

        loss_a = metric(img_pred)
        loss_b = metric(img_gt)
        loss_list_a.append(loss_a.item())
        loss_list_b.append(loss_b.item())

    return (sum(loss_list_a) / len(loss_list_a), sum(loss_list_b) / len(loss_list_b)), (loss_list_a, loss_list_b)


def eval_metric(fake_path, real_path, metric):
    fake_list = []
    real_list = []

    # print(len(os.listdir(fake_path)))

    for img_name in os.listdir(fake_path):
        if img_name.endswith('.png'):
            fake_list.append(osp.join(fake_path, img_name))
            real_list.append(osp.join(real_path, img_name))

    if metric.__name__ not in metric_no_ref:
        print(metric.__name__)
        mean_loss, loss_list = quality_metric(fake_list, real_list, metric)
    else:
        print(metric.__name__)
        mean_loss, loss_list = quality_metric_no_ref(fake_list, real_list, metric)

    # print(mean_loss)  # 0.6306624141335487
    return mean_loss, loss_list


def mind(x, y, *args):
    """
    Deep Image Structure and Texture Similarity metric.
    """
    if torch.cuda.is_available():
        x = x.to('cuda:0')
        y = y.to('cuda:0')
        loss = MINDLoss(*args).to('cuda:0')
    else:
        loss = MINDLoss(*args)
    return 100 * loss(x, y)


def mutual_info(x, y, *args):
    """
    Deep Image Structure and Texture Similarity metric.
    """
    if torch.cuda.is_available():
        x = x.to('cuda:0')
        y = y.to('cuda:0')
        loss = MILoss(*args).to('cuda:0')
    else:
        loss = MILoss(*args)
    return loss(x, y)


def dists(x, y, *args):
    """
    Deep Image Structure and Texture Similarity metric.
    """
    if torch.cuda.is_available():
        x = x.to('cuda:0')
        y = y.to('cuda:0')
    loss = piq.DISTS(*args)
    return loss(x, y)


def lpips(x, y, *args):
    """
    Deep Image Structure and Texture Similarity metric.
    """
    if torch.cuda.is_available():
        x = x.to('cuda:0')
        y = y.to('cuda:0')
    loss = piq.LPIPS(*args)
    return loss(x, y)


def pieapp(x, y, *args):
    """
    Deep Image Structure and Texture Similarity metric.
    """
    if torch.cuda.is_available():
        x = x.to('cuda:0')
        y = y.to('cuda:0')
    loss = piq.PieAPP(*args)
    return loss(x, y)


def basic():
    import os

    # train
    # with batch_size = 4, take ~50 hrs
    # sh = "python train.py --dataroot ./datasets/horse2zebra --name horse2zebra --model cycle_gan --batch_size 2"

    dataset = "brain_mr2ct"  # mr2ct
    # dataset = "retina2vessel"  # mr2ct
    # dataset = "oneshot"  # mr2ct
    # dataset = "horse2zebra"  # mr2ct

    name = "paired"

    sh = f"python train.py --dataroot ./datasets/{dataset} --name {dataset}_{name}"
    paras = [
        " --model pix2pix",  # cycle_gan,
        # " --direction BtoA",
        " --n_epochs 100",
        " --n_epochs_decay 100",
        " --gan_mode lsgan",  # vanilla, lsgan, lsgan+mind
        " --batch_size 4",
        # " --netG unet_256",
        # " --ngf 96",
        " --single_D True",
        # " --single_G True",
        # " --netD pixel",  # pixel
        # " --n_layers_D 5",
        # " --lambda_identity 0 --display_ncols 3",

        # " --cycle_loss SSMI",
        # " --aug_policy color,cutout",  # color,translation,cutout
        # " --aug_threshold 0.5",  # 1 - p
        # " --preprocess scale_width_and_crop --num_threads 0",
        # " --align_mode DISTS",
        # " --lambda_DISTS 0.5",

        # " --align_mode MIND",
        # " --lambda_MIND 0",

        # " --align_mode MI",
        # " --lambda_MI 0.1",

        # " --lambda_A 10",
        # " --lambda_B 10",
        " --save_epoch_freq 50",
        " --load_size 256",
        " --crop_size 256",
        " --input_nc 1",
        " --output_nc 1",
    ]

    for x in paras:
        sh += x

    # os.system(sh)

    # test
    # sh = "python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout"
    sh = f"python test.py --name {dataset}_{name} --model test --no_dropout"
    #
    paras = [
        f" --dataroot datasets/{dataset}",
        " --model pix2pix",  # cycle_gan,
        # " --direction BtoA",
        # " --netG unet_256",
        " --load_size 256",
        # " --crop_size 512",
        " --num_test 200"
        " --input_nc 1",
        " --output_nc 1",
    ]
    #
    for x in paras:
        sh += x

    os.system(sh)

    from piq import ssim, psnr

    fake_path = f"results/brain_mr2ct_paired/test_latest/images"
    real_path = f"Z:/GAN/pytorch-CycleGAN-and-pix2pix-master/datasets/brain_mr2ct/allB"

    eval_metric(fake_path, real_path, ssim)


class Pix2Pix:
    def __init__(self, dataset, name, batch_size, gan_mode='lsgan', continue_train=False):
        self.dataset = dataset  # mr2ct
        self.name = name
        self.batch_size = batch_size
        self.gan_mode = gan_mode
        self.continue_train = continue_train

    def train(self):

        sh = f"python train.py --dataroot ./datasets/{self.dataset} --name {self.dataset}_{self.name}"
        paras = [
            " --model pix2pix",  # cycle_gan,
            " --n_epochs 100",
            " --n_epochs_decay 100",
            f" --gan_mode {self.gan_mode}",  # vanilla, lsgan
            f" --batch_size {self.batch_size}",
            f" --continue_train" if self.continue_train else "",

            " --save_epoch_freq 50",
            " --load_size 256",
            " --crop_size 256",
            " --input_nc 1",
            " --output_nc 1",
        ]

        for x in paras:
            sh += x

        os.system(sh)

    def test(self):
        # test
        # sh = "python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout"
        sh = f"python test.py --name {self.dataset}_{self.name} --model test --no_dropout"
        #
        paras = [
            f" --dataroot datasets/{self.dataset}",
            " --model pix2pix",  # cycle_gan,
            " --load_size 256",
            " --crop_size 256",
            " --num_test 1000"
            " --input_nc 1",
            " --output_nc 1",
        ]
        #
        for x in paras:
            sh += x

        os.system(sh)

    def eval(self, metric):

        fake_path = f"results/{self.dataset}_{self.name}/test_latest/images"
        real_path = f"Z:/GAN/pytorch-CycleGAN-and-pix2pix-master/datasets/{self.dataset}/allB"

        eval_metric(fake_path=fake_path, real_path=real_path, metric=metric)


class CycleGAN:
    def __init__(self, dataset, name, model="dcl", load_size=512, dataset_mode='unaligned', netG='resnet_9blocks',
                 input_nc=1, output_nc=1, dataroot=None, gpu_ids='0',
                 extra=''):
        self.dataset = dataset  # mr2ct
        self.name = name

        self.model = model
        self.netG = netG
        self.load_size = load_size
        self.dataset_mode = dataset_mode
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.dataroot = dataroot if dataroot is not None else "./datasets"

        self.shared_args = [
            f" --netG {netG}",
            f" --dataset_mode {self.dataset_mode}",
            f" --dataroot {self.dataroot}/{self.dataset}",
            f" --gpu_ids {gpu_ids}",
            extra
        ]

    def train(self, batch_size=2, n_epochs=100, n_epochs_decay=100, lambda_identity=0.5,
              extra='',
              continue_train=False):
        sh = f"python train.py --dataroot ./datasets/{self.dataset} --name {self.dataset}_{self.name}"
        paras = [
            f" --model {self.model}",  # cycle_gan, simdcl, cut, fastcut
            # " --direction BtoA",
            f" --lambda_identity {lambda_identity}",
            f" --n_epochs {n_epochs}",
            f" --n_epochs_decay {n_epochs_decay}",
            f" --gan_mode hinge",  # vanilla, lsgan, lsgan+mind
            f" --batch_size {batch_size}",

            f" --load_size {self.load_size}",
            " --crop_size 256",
            f" --input_nc {self.input_nc} --output_nc {self.output_nc}",

            " --num_threads 0",
            " --continue_train" if continue_train else "",
            # f" --lambda_identity {lambda_identity}",
            # " --netG unet_256",
            # " --ngf 96",
            # " --netD pixel",  # pixel
            # " --n_layers_D 5",
            # " --lambda_identity 0 --display_ncols 3" if self.no_identity else "",

            # " --lambda_A 10",
            # " --lambda_B 10",
            " --save_epoch_freq 50",
            " --display_ncols 3",
            extra
        ]

        for x in chain(self.shared_args, paras):
            sh += x

        os.system(sh)

    def test(self, script='test', num_test=500, load_size=256, crop_size=256, epoch='latest', extra=''):
        # test
        # sh = "python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout"
        sh = f"python {script}.py --name {self.dataset}_{self.name} --phase test --no_dropout"
        #
        paras = [
            f" --dataroot {self.dataroot}/{self.dataset}",
            f" --dataset_mode {self.dataset_mode}",
            f" --model {self.model}",
            f" --load_size {load_size}",
            f" --crop_size {crop_size}",
            f" --num_test {num_test}",
            f" --epoch {epoch}"
            " --eval",
            extra
        ]
        #
        for x in chain(self.shared_args, paras):
            sh += x
        print(sh)
        os.system(sh)

    @torch.no_grad()
    def eval_with_return(self, metric_list, testB=False, phase='test', epoch='latest'):

        # test A: A -> B
        realB_path = f"results/{self.dataset}_{self.name}/{phase}_{epoch}/images/real_B"
        fakeB_path = f"results/{self.dataset}_{self.name}/{phase}_{epoch}/images/fake_B"
        # test B: B -> A
        realA_path = f"results/{self.dataset}_{self.name}/{phase}_{epoch}/images/real_A"
        fakeA_path = f"results/{self.dataset}_{self.name}/{phase}_{epoch}/images/fake_A"

        mean_loss_A, loss_list_A = [], []
        mean_loss_B, loss_list_B = [], []
        exp_row_A = [self.name]
        exp_row_B = [self.name]

        for metric in metric_list:

            mean_loss, loss_list = eval_metric(fake_path=fakeB_path, real_path=realB_path, metric=eval(metric))
            if isinstance(mean_loss, float):
                mean_loss = round(mean_loss, 4)
            if isinstance(mean_loss, tuple):
                mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            mean_loss_A.append(mean_loss)
            loss_list_A.append(loss_list)

            if testB:
                mean_loss, loss_list = eval_metric(fake_path=fakeA_path, real_path=realA_path, metric=eval(metric))
                if isinstance(mean_loss, float):
                    mean_loss = round(mean_loss, 4)
                if isinstance(mean_loss, tuple):
                    mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            elif metric in metric_no_ref:
                mean_loss, loss_list = ('-', '-'), '-'
            else:
                mean_loss, loss_list = '-', '-'
            mean_loss_B.append(mean_loss)
            loss_list_B.append(loss_list)

        # table = PrettyTable(title=self.dataset, field_names=['metric', 'testA', 'testB'])
        table = PrettyTable(field_names=['metric', 'testA', 'testB'])

        for idx, metric in enumerate(metric_list):
            if metric not in metric_no_ref:
                table.add_row([metric, mean_loss_A[idx], mean_loss_B[idx]])
                exp_row_A.append(mean_loss_A[idx])
                exp_row_B.append(mean_loss_B[idx])
            else:
                table.add_row([metric,
                               f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})",
                               f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})"])
                exp_row_A.append(f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})")
                exp_row_B.append(f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})")

        print(-mean_loss_A[0])

    @torch.no_grad()
    def dists_with_return(self, phase='test', epoch='latest'):

        # test A: A -> B
        realB_path = f"results/{self.dataset}_{self.name}/{phase}_{epoch}/images/real_B"
        fakeB_path = f"results/{self.dataset}_{self.name}/{phase}_{epoch}/images/fake_B"
        # test B: B -> A
        realA_path = f"results/{self.dataset}_{self.name}/{phase}_{epoch}/images/real_A"
        fakeA_path = f"results/{self.dataset}_{self.name}/{phase}_{epoch}/images/fake_A"

        sh = f"python Z:/CodingHere/GAN/metric/DISTS-master/DISTS_pytorch/DISTS_folder_pt.py --ref {realB_path} --dist {fakeB_path}"
        # print(os.popen(sh).readlines())
        eval_score = float(os.popen(sh).readlines()[-2].strip('/n').strip())

        print(-eval_score)

        return -eval_score

    @torch.no_grad()
    def eval(self, metric_list, testB=False):

        # test A: A -> B
        realB_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_B"
        fakeB_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_B"
        # test B: B -> A
        realA_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_A"
        fakeA_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_A"

        help_dct = {
            "psnr": "Peak Signal-to-Noise Ratio",
            "ssmi": "Structural Similarity",
            "multi_scale_ssim": "Multi-Scale Structural Similarity",
            "vif_p": "Visual Information Fidelity",
            "fsim": "Feature Similarity Index Measure",
            "gmsd": "Gradient Magnitude Similarity Deviation",
            "multi_scale_gmsd": "Multi-Scale Gradient Magnitude Similarity Deviation",
            "haarpsi": "Haar Perceptual Similarity Index",
            "mdsi": "Mean Deviation Similarity Index",
        }

        mean_loss_A, loss_list_A = [], []
        mean_loss_B, loss_list_B = [], []
        exp_row_A = [self.name]
        exp_row_B = [self.name]

        for metric in metric_list:

            mean_loss, loss_list = eval_metric(fake_path=fakeB_path, real_path=realB_path, metric=eval(metric))
            if isinstance(mean_loss, float):
                mean_loss = round(mean_loss, 4)
            if isinstance(mean_loss, tuple):
                mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            mean_loss_A.append(mean_loss)
            loss_list_A.append(loss_list)

            if testB:
                mean_loss, loss_list = eval_metric(fake_path=fakeA_path, real_path=realA_path, metric=eval(metric))
                if isinstance(mean_loss, float):
                    mean_loss = round(mean_loss, 4)
                if isinstance(mean_loss, tuple):
                    mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            elif metric in metric_no_ref:
                mean_loss, loss_list = ('-', '-'), '-'
            else:
                mean_loss, loss_list = '-', '-'
            mean_loss_B.append(mean_loss)
            loss_list_B.append(loss_list)

        # table = PrettyTable(title=self.dataset, field_names=['metric', 'testA', 'testB'])
        table = PrettyTable(field_names=['metric', 'testA', 'testB'])

        for idx, metric in enumerate(metric_list):
            if metric not in metric_no_ref:
                table.add_row([metric, mean_loss_A[idx], mean_loss_B[idx]])
                exp_row_A.append(mean_loss_A[idx])
                exp_row_B.append(mean_loss_B[idx])
            else:
                table.add_row([metric,
                               f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})",
                               f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})"])
                exp_row_A.append(f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})")
                exp_row_B.append(f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})")

        print(f"evaluating on {self.dataset}, totally {len(os.listdir(realA_path))} samples.")
        print(table)

        print("PSNR: in [20, 40], larger is the better")
        print("SSIM: in [0, 1], larger is the better")

        rowsA.append(exp_row_A)
        rowsB.append(exp_row_B)

        return exp_row_A, exp_row_B

    @torch.no_grad()
    def eval_spos(self, metric_list, choice_spos, testB=False):

        # test A: A -> B
        realB_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_B/{choice_spos}"
        fakeB_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_B/{choice_spos}"
        # test B: B -> A
        realA_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_A/{choice_spos}"
        fakeA_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_A/{choice_spos}"

        help_dct = {
            "psnr": "Peak Signal-to-Noise Ratio",
            "ssmi": "Structural Similarity",
            "multi_scale_ssim": "Multi-Scale Structural Similarity",
            "vif_p": "Visual Information Fidelity",
            "fsim": "Feature Similarity Index Measure",
            "gmsd": "Gradient Magnitude Similarity Deviation",
            "multi_scale_gmsd": "Multi-Scale Gradient Magnitude Similarity Deviation",
            "haarpsi": "Haar Perceptual Similarity Index",
            "mdsi": "Mean Deviation Similarity Index",
            "ssmi": "Similarity",
        }

        mean_loss_A, loss_list_A = [], []
        mean_loss_B, loss_list_B = [], []
        exp_row_A = [choice_spos]
        exp_row_B = [choice_spos]

        for metric in metric_list:

            mean_loss, loss_list = eval_metric(fake_path=fakeB_path, real_path=realB_path, metric=eval(metric))
            if isinstance(mean_loss, float):
                mean_loss = round(mean_loss, 4)
            if isinstance(mean_loss, tuple):
                mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            mean_loss_A.append(mean_loss)
            loss_list_A.append(loss_list)

            if testB:
                mean_loss, loss_list = eval_metric(fake_path=fakeA_path, real_path=realA_path, metric=eval(metric))
                if isinstance(mean_loss, float):
                    mean_loss = round(mean_loss, 4)
                if isinstance(mean_loss, tuple):
                    mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            elif metric in metric_no_ref:
                mean_loss, loss_list = ('-', '-'), '-'
            else:
                mean_loss, loss_list = '-', '-'
            mean_loss_B.append(mean_loss)
            loss_list_B.append(loss_list)

        # table = PrettyTable(title=self.dataset, field_names=['metric', 'testA', 'testB'])
        table = PrettyTable(field_names=['metric', 'testA', 'testB'])

        for idx, metric in enumerate(metric_list):
            if metric not in metric_no_ref:
                table.add_row([metric, mean_loss_A[idx], mean_loss_B[idx]])
                exp_row_A.append(mean_loss_A[idx])
                exp_row_B.append(mean_loss_B[idx])
            else:
                table.add_row([metric,
                               f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})",
                               f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})"])
                exp_row_A.append(f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})")
                exp_row_B.append(f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})")

        print(f"evaluating on {self.dataset}, totally {len(os.listdir(realA_path))} samples.")
        print(table)

        print("PSNR: in [20, 40], larger is the better")
        print("SSIM: in [0, 1], larger is the better")

        rowsA.append(exp_row_A)
        rowsB.append(exp_row_B)

        return mean_loss_A, mean_loss_B

    def fid(self, epoch='latest'):

        sh = f"python -m pytorch_fid" \
             f" ./results/{self.dataset}_{self.name}/test_{epoch}/images/fake_B" \
             f" ./results/{self.dataset}_{self.name}/test_{epoch}/images/real_B"
        print(f"evaluating FID on ./results/{self.dataset}_{self.name}/test_{epoch}/images/fake_B")
        os.system(sh)

    def fid_with_return(self, phase='test', epoch='latest'):
        sh = f"python -m pytorch_fid --device cuda:0 --num-workers 0" \
             f" ./results/{self.dataset}_{self.name}/{phase}_{epoch}/images/fake_B" \
             f" ./results/{self.dataset}_{self.name}/{phase}_{epoch}/images/real_B"
        print(f"evaluating FID on ./results/{self.dataset}_{self.name}/{phase}_{epoch}/images/fake_B")
        # print(os.popen(sh).readlines())
        eval_score = round(float(os.popen(sh).readlines()[-2].strip('/n').split(':')[-1].strip()), 4)
        print('FID: ', eval_score)
        return eval_score

    def eval_arch_searched(self, eval_num=10, log='log'):
        table = PrettyTable(title=f"top {eval_num} arch", field_names=['candidates', 'train-FID', 'test-FID'])
        info = torch.load(f'{log}/checkpoint.pth.tar')

        vis_list = sorted(
            [(k, info['vis_dict'][k].get('err')) for k in info['vis_dict'].keys() if
             info['vis_dict'][k].get('err') is not None],
            key=lambda x: x[1])
        for cand, fid in vis_list[:eval_num]:
            choice = str(list(cand)).replace(' ', '')
            self.test(extra=f' --choice_G_A {choice} --batch_size 6 --direction AtoB')
            table.add_row([str(cand), fid, self.fid_with_return()])
        with open(f'checkpoints/{self.dataset}_{self.name}/arch_searched.txt', mode='a') as f:
            f.write(str(table))
        print(table)

    def eval_arch_searched_dists(self, eval_num=10, log='log'):
        table = PrettyTable(title=f"top {eval_num} arch", field_names=['candidates', 'train-FID', 'test-FID'])
        info = torch.load(f'{log}/checkpoint.pth.tar')

        vis_list = sorted(
            [(k, info['vis_dict'][k].get('err')) for k in info['vis_dict'].keys() if
             info['vis_dict'][k].get('err') is not None],
            key=lambda x: x[1])
        for cand, fid in vis_list[:eval_num]:
            choice = str(list(cand)).replace(' ', '')
            self.test(extra=f' --choice_G_B {choice} --batch_size 6 --direction BtoA')
            table.add_row([str(cand), fid, self.dists_with_return()])
        with open(f'checkpoints/{self.dataset}_{self.name}/arch_searched.txt', mode='a') as f:
            f.write(str(table))
        print(table)


class Experiment:
    def __init__(self, dataset, name, model="dcl", load_size=256, netG="resnet_9blocks", gpu_ids='0',
                 input_nc=1, output_nc=1, dataroot=None, extra="",
                 dataset_mode='unaligned'):
        self.dataset = dataset  # mr2ct
        self.name = name
        self.model = model
        self.gpu_ids = gpu_ids
        self.load_size = load_size
        self.dataset_mode = dataset_mode
        self.dataroot = dataroot if dataroot is not None else "/home/yht/Casit/Datasets/ez/datasets"

        self.shared_args = [
            f" --input_nc {input_nc} --output_nc {output_nc}",
            f" --netG {netG}",
            f" --gpu_ids {self.gpu_ids}",
            extra,
        ]

    def train(self, batch_size=2, n_epochs=100, n_epochs_decay=100, lambda_identity=0.5, input_nc=1, output_nc=1,
              nce_idt=True, display_ncols=3, netD='basic', n_layers_D=3,
              continue_train=False, script_name='train', extra=""):
        start = time.time()
        sh = f"python {script_name}.py --dataroot {self.dataroot}/{self.dataset} --name {self.dataset}_{self.name}"
        paras = [
            f" --model {self.model}",  # cycle_gan, simdcl, cut, fastcut
            # " --direction BtoA",
            f" --netD {netD}",
            f" --n_layers_D {n_layers_D}",
            f" --nce_idt {nce_idt}",
            f" --n_epochs {n_epochs}",
            f" --n_epochs_decay {n_epochs_decay}",
            f" --gan_mode hinge",  # vanilla, lsgan, lsgan+mind
            f" --dataset_mode {self.dataset_mode}",
            f" --batch_size {batch_size}",
            " --display_port 8098",
            f" --load_size {self.load_size}",
            " --crop_size 256",
            " --save_epoch_freq 10",
            " --num_threads 8",
            " --continue_train" if continue_train else "",
            # f" --lambda_identity {lambda_identity}",
            # " --netG unet_256",
            # " --ngf 96",
            # " --netD pixel",  # pixel
            # " --n_layers_D 5",
            # " --lambda_identity 0 --display_ncols 3" if self.no_identity else "",

            # " --lambda_A 10",
            # " --lambda_B 10",
            f" --display_ncols {display_ncols}",
        ]

        for x in chain(self.shared_args, paras):
            sh += x

        sh += extra
        print(sh)
        os.system(sh)

        m, s = divmod(time.time() - start, 60)
        h, m = divmod(m, 60)
        print(f"Training completed in {h:.1f}hrs, {m:.1f}min, {s:.1f}sec!")

    def test(self, script='test', num_test=500, load_size=256, crop_size=256, epoch='latest', eval=True, extra=''):
        start = time.time()
        # sh = "python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout"
        sh = f"python {script}.py --name {self.dataset}_{self.name} --phase test --no_dropout"
        #
        paras = [
            f" --dataroot {self.dataroot}/{self.dataset}",
            f" --dataset_mode {self.dataset_mode}",
            f" --model {self.model}",
            f" --load_size {load_size}",
            f" --crop_size {crop_size}",
            f" --num_test {num_test}",
            f" --epoch {epoch}"
            " --eval" if eval else "",
            extra
        ]
        #
        for x in chain(self.shared_args, paras):
            sh += x
        print(sh)
        os.system(sh)

        m, s = divmod(time.time() - start, 60)
        h, m = divmod(m, 60)
        print(f"Testing completed in {h:.1f}hrs, {m:.1f}min, {s:.1f}sec!")

    @torch.no_grad()
    def eval(self, metric_list, testB=False):
        start = time.time()
        # test A: A -> B
        realB_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_B"
        fakeB_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_B"
        # test B: B -> A
        realA_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_A"
        fakeA_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_A"

        help_dct = {
            "psnr": "Peak Signal-to-Noise Ratio",
            "ssmi": "Structural Similarity",
            "multi_scale_ssim": "Multi-Scale Structural Similarity",
            "vif_p": "Visual Information Fidelity",
            "fsim": "Feature Similarity Index Measure",
            "gmsd": "Gradient Magnitude Similarity Deviation",
            "multi_scale_gmsd": "Multi-Scale Gradient Magnitude Similarity Deviation",
            "haarpsi": "Haar Perceptual Similarity Index",
            "mdsi": "Mean Deviation Similarity Index",
            "ssmi": "Similarity",
        }

        mean_loss_A, loss_list_A = [], []
        mean_loss_B, loss_list_B = [], []
        exp_row_A = [self.name]
        exp_row_B = [self.name]

        for metric in metric_list:

            mean_loss, loss_list = eval_metric(fake_path=fakeB_path, real_path=realB_path, metric=eval(metric))
            if isinstance(mean_loss, float):
                mean_loss = round(mean_loss, 4)
            if isinstance(mean_loss, tuple):
                mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            mean_loss_A.append(mean_loss)
            loss_list_A.append(loss_list)

            if testB:
                mean_loss, loss_list = eval_metric(fake_path=fakeA_path, real_path=realA_path, metric=eval(metric))
                if isinstance(mean_loss, float):
                    mean_loss = round(mean_loss, 4)
                if isinstance(mean_loss, tuple):
                    mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            elif metric in metric_no_ref:
                mean_loss, loss_list = ('-', '-'), '-'
            else:
                mean_loss, loss_list = '-', '-'
            mean_loss_B.append(mean_loss)
            loss_list_B.append(loss_list)

        # table = PrettyTable(title=self.dataset, field_names=['metric', 'testA', 'testB'])
        table = PrettyTable(field_names=['metric', 'testA', 'testB'])

        for idx, metric in enumerate(metric_list):
            if metric not in metric_no_ref:
                table.add_row([metric, mean_loss_A[idx], mean_loss_B[idx]])
                exp_row_A.append(mean_loss_A[idx])
                exp_row_B.append(mean_loss_B[idx])
            else:
                table.add_row([metric,
                               f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})",
                               f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})"])
                exp_row_A.append(f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})")
                exp_row_B.append(f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})")

        print(f"evaluating on {self.dataset}, totally {len(os.listdir(realA_path))} samples.")
        print(table)

        print("PSNR: in [20, 40], larger is the better")
        print("SSIM: in [0, 1], larger is the better")

        rowsA.append(exp_row_A)
        rowsB.append(exp_row_B)

        m, s = divmod(time.time() - start, 60)
        h, m = divmod(m, 60)
        print(f"Evaluation completed in {h:.1f}hrs, {m:.1f}min, {s:.1f}sec!")

        return exp_row_A, exp_row_B

    @torch.no_grad()
    def eval_with_return(self, metric, lower_is_better=False):

        # test A: A -> B
        realB_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_B"
        fakeB_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_B"

        mean_loss, loss_list = eval_metric(fake_path=fakeB_path, real_path=realB_path, metric=eval(metric))
        if isinstance(mean_loss, float):
            mean_loss = round(mean_loss, 4)
        if isinstance(mean_loss, tuple):
            mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))
        if lower_is_better:
            return mean_loss
        else:
            return -mean_loss

    @torch.no_grad()
    def eval_cityscapes(self, epoch='latest'):
        import os

        data_folder = f"{self.dataset}_{self.name}"
        out_dir = f'/home/yht/Casit/ez/CUT-Former/results/{data_folder}/test_{epoch}/images/drn_results'
        ckpt_path = "/home/yht/Casit/ez/drn-master/drn_d_22.pth.tar"
        sh = f"python /home/yht/Casit/ez/drn-master/segment.py test -d {data_folder} -o {out_dir}" \
             f" -c 19 --arch drn_d_22 --batch-size 1" \
             f" --resume {ckpt_path} --phase val --with-gt"

        os.system(sh)

    def swd(self, epoch='latest'):
        sh = f"python util/SWD/cal_sliced_wasserstein.py" \
             f" ./results/{self.dataset}_{self.name}/test_{epoch}/images/fake_B" \
             f" ./results/{self.dataset}_{self.name}/test_{epoch}/images/real_B"
        print(sh)
        print(f"evaluating SWD on ./results/{self.dataset}_{self.name}/test_{epoch}/images/fake_B")
        os.system(sh)

    def swd_with_return(self, phase='test', epoch='latest'):
        sh = f"python util/SWD/cal_sliced_wasserstein.py" \
             f" ./results/{self.dataset}_{self.name}/{phase}_{epoch}/images/fake_B" \
             f" ./results/{self.dataset}_{self.name}/{phase}_{epoch}/images/real_B"
        print(f"evaluating SWD on ./results/{self.dataset}_{self.name}/{phase}_{epoch}/images/fake_B")
        # print(os.popen(sh).readlines())
        eval_score = round(float(os.popen(sh).readlines()[-1].strip('/n').split(':')[-1].strip()), 4)
        print('SWD: ', eval_score)
        return eval_score

    @torch.no_grad()
    def eval_spos(self, metric_list, choice_spos, testB=False):

        # test A: A -> B
        realB_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_B/{choice_spos}"
        fakeB_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_B/{choice_spos}"
        # test B: B -> A
        realA_path = f"results/{self.dataset}_{self.name}/test_latest/images/real_A/{choice_spos}"
        fakeA_path = f"results/{self.dataset}_{self.name}/test_latest/images/fake_A/{choice_spos}"

        help_dct = {
            "psnr": "Peak Signal-to-Noise Ratio",
            "ssmi": "Structural Similarity",
            "multi_scale_ssim": "Multi-Scale Structural Similarity",
            "vif_p": "Visual Information Fidelity",
            "fsim": "Feature Similarity Index Measure",
            "gmsd": "Gradient Magnitude Similarity Deviation",
            "multi_scale_gmsd": "Multi-Scale Gradient Magnitude Similarity Deviation",
            "haarpsi": "Haar Perceptual Similarity Index",
            "mdsi": "Mean Deviation Similarity Index",
        }

        mean_loss_A, loss_list_A = [], []
        mean_loss_B, loss_list_B = [], []
        exp_row_A = [choice_spos]
        exp_row_B = [choice_spos]

        for metric in metric_list:

            mean_loss, loss_list = eval_metric(fake_path=fakeB_path, real_path=realB_path, metric=eval(metric))
            if isinstance(mean_loss, float):
                mean_loss = round(mean_loss, 4)
            if isinstance(mean_loss, tuple):
                mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            mean_loss_A.append(mean_loss)
            loss_list_A.append(loss_list)

            if testB:
                mean_loss, loss_list = eval_metric(fake_path=fakeA_path, real_path=realA_path, metric=eval(metric))
                if isinstance(mean_loss, float):
                    mean_loss = round(mean_loss, 4)
                if isinstance(mean_loss, tuple):
                    mean_loss = (round(mean_loss[0], 4), round(mean_loss[1], 4))

            elif metric in metric_no_ref:
                mean_loss, loss_list = ('-', '-'), '-'
            else:
                mean_loss, loss_list = '-', '-'
            mean_loss_B.append(mean_loss)
            loss_list_B.append(loss_list)

        # table = PrettyTable(title=self.dataset, field_names=['metric', 'testA', 'testB'])
        table = PrettyTable(field_names=['metric', 'testA', 'testB'])

        for idx, metric in enumerate(metric_list):
            if metric not in metric_no_ref:
                table.add_row([metric, mean_loss_A[idx], mean_loss_B[idx]])
                exp_row_A.append(mean_loss_A[idx])
                exp_row_B.append(mean_loss_B[idx])
            else:
                table.add_row([metric,
                               f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})",
                               f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})"])
                exp_row_A.append(f"{mean_loss_A[idx][0]} ({mean_loss_A[idx][1]})")
                exp_row_B.append(f"{mean_loss_B[idx][0]} ({mean_loss_B[idx][1]})")

        print(f"evaluating on {self.dataset}, totally {len(os.listdir(realA_path))} samples.")
        print(table)

        print("PSNR: in [20, 40], larger is the better")
        print("SSIM: in [0, 1], larger is the better")

        rowsA.append(exp_row_A)
        rowsB.append(exp_row_B)

        return mean_loss_A, mean_loss_B

    def fid(self, epoch='latest'):
        if self.dataset == 'cityscapes':
            sh = f"python -m pytorch_fid --device {self.gpu_ids}" \
                 f" /home/yht/Casit/ez/datasets/cityscapes/testA" \
                 f" ./results/{self.dataset}_{self.name}/test_{epoch}/images/fake_B"
        # else:
        #     sh = f"python -m pytorch_fid" \
        #          f" /home/yht/Casit/ez/datasets/{self.dataset}/testB" \
        #          f" ./results/{self.dataset}_{self.name}/test_{epoch}/images/fake_B"
        else:
            sh = f"python -m pytorch_fid --device cuda:{self.gpu_ids}" \
                 f" ./results/{self.dataset}_{self.name}/test_{epoch}/images/real_B" \
                 f" ./results/{self.dataset}_{self.name}/test_{epoch}/images/fake_B"
        print(sh)
        print(f"evaluating FID on ./results/{self.dataset}_{self.name}/test_{epoch}/images/fake_B")
        os.system(sh)

    def fid_with_return(self, phase='test', epoch='latest'):
        sh = f"python -m pytorch_fid --device cuda:{self.gpu_ids} --num-workers 0" \
             f" ./results/{self.dataset}_{self.name}/{phase}_{epoch}/images/fake_B" \
             f" ./results/{self.dataset}_{self.name}/{phase}_{epoch}/images/real_B"
        print(f"evaluating FID on ./results/{self.dataset}_{self.name}/{phase}_{epoch}/images/fake_B")
        # print(os.popen(sh).readlines())
        eval_score = round(float(os.popen(sh).readlines()[-1].strip('/n').split(':')[-1].strip()), 4)
        print('FID: ', eval_score)
        return eval_score

    def traverse_fid(self, ):
        table = PrettyTable(title=f"{self.dataset}_{self.name}", field_names=['epoch_num', 'FID'])
        epoches = sorted([x.replace('_net_G.pth', '') for x in
                          os.listdir(f"/home/yht/Casit/ez/EnCo-pytorch/checkpoints/{self.dataset}_{self.name}")
                          if '_net_G.pth' in x and 'latest' not in x],
                         key=lambda x: int(x))
        for epoch in epoches:
            self.test(epoch=epoch, extra=' --batch_size 8')
            fid = self.fid_with_return(epoch=epoch)
            table.add_row([epoch, fid])
        with open(f'/home/yht/Casit/ez/EnCo-pytorch/checkpoints/{self.dataset}_{self.name}/traverse_fid.txt',
                  mode='a') as f:
            f.write(str(table))
            print(
                f'writing into /home/yht/Casit/ez/EnCo-pytorch/checkpoints/{self.dataset}_{self.name}/traverse_fid.txt')
        print(table)

    def eval_arch_searched(self, eval_num=10, log='log'):
        table = PrettyTable(title=f"top {eval_num} arch", field_names=['candidates', 'train-FID', 'test-FID'])
        info = torch.load(f'{log}/checkpoint.pth.tar')

        vis_list = sorted(
            [(k, info['vis_dict'][k].get('err')) for k in info['vis_dict'].keys() if
             info['vis_dict'][k].get('err') is not None],
            key=lambda x: x[1])
        for cand, fid in vis_list[:eval_num]:
            choice = str(list(cand)).replace(' ', '')
            self.test(extra=f' --choice_spos {choice} --batch_size 6')
            table.add_row([str(cand), fid, self.fid_with_return()])
        with open(f'checkpoints/{self.dataset}_{self.name}/arch_searched.txt', mode='a') as f:
            f.write(str(table))
        print(table)


def show_results(metric_list, rowsA, rowsB, testB=True):
    # table = PrettyTable(title=exp.dataset, field_names=['metric', 'testA', 'testB'])
    field_names = ['method']
    for metric in metric_list:
        field_names.append(metric)
    tableA = PrettyTable(title="testA", field_names=field_names)
    tableB = PrettyTable(title="testB", field_names=field_names)

    for rowA, rowB in zip(rowsA, rowsB):
        tableA.add_row(rowA)
        tableB.add_row(rowB)

    print(tableA)
    if testB:
        print(tableB)


def baseline():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'vsi', 'total_variation']
    # metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    # 1546 sec, ~0.5h
    exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cyclegan_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="dcl", name="dcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="simdcl", name="simdcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False,)
    #           # extra=" --nce_includes_all_negatives_from_minibatch True")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    show_results(metric_list, rowsA, rowsB)


def enco_horse():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'vsi', 'total_variation']

    exp = Experiment(dataset="horse2zebra_curated", model="encov2",
                     name="encov2_3,10,13,18,21,28_ttur_warmup_ep400_id1",
                     load_size=286, gpu_ids='0', input_nc=3, output_nc=3,
                     dataroot="/home/yht/Casit/Datasets/ez/datasets", )
    exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False, netD='sa',
              extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,10,13,18,21,28 --display_freq 100'
                    ' --stop_idt_epochs 400 --gan_mode ralsgan'
                    ' --lambda_IDT 5 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient True'
                    ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --flip_equivariance True'
                    ' --num_patches 256')
    exp.test(extra=' --batch_size 8', epoch='latest')
    exp.fid(epoch='latest')  # FID:
    exp.traverse_fid()

    exp = Experiment(dataset="horse2zebra", model="enco", name="enco_next_3,7,11,13,18,20,24,28_ttur_warmup_ep400_id10",
                     load_size=286, gpu_ids='0', input_nc=3, output_nc=3,
                     dataroot="/home/yht/Casit/Datasets/ez/datasets",
                     netG='next')
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False, netD='sa',
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,11,13,18,20,24,28 --display_freq 100'
    #                 ' --stop_idt_epochs 400'
    #                 ' --lambda_IDT 5 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient True --gan_mode lsgan'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --flip_equivariance True'
    #                 ' --num_patches 256')
    # exp.test(extra=' --batch_size 8', epoch='latest')
    # exp.fid(epoch='latest')  # FID:
    # exp.traverse_fid()
    # |          epoch_num           |            FID            |
    # +------------------------------+---------------------------+
    # |              10              |          287.6622         |
    # |              20              |          176.5636         |
    # |              30              |          124.7092         |
    # |              40              |          82.6716          |
    # |              50              |          163.8198         |
    # |              60              |          117.8704         |
    # |              70              |          107.923          |
    # |              80              |          81.4565          |
    # |              90              |          87.6456          |
    # |             100              |          88.6359          |
    # |             110              |          139.6939         |
    # |             120              |          73.4437          |
    # |             130              |           78.856          |
    # |             140              |          72.8921          |
    # |             150              |          76.9896          |
    # |             160              |          66.8545          |
    # |             170              |          64.2294          |
    # |             180              |           80.274          |
    # |             190              |          106.1818         |
    # |             200              |          110.447          |
    # |             210              |          57.1118          |
    # |             220              |          77.8433          |
    # |             230              |          73.5659          |
    # |             240              |          70.3128          |
    # |             250              |          72.5871          |
    # |             260              |          54.5832          |
    # |             270              |          75.5198          |
    # |             280              |          61.6277          |
    # |             290              |           61.737          |
    # |             300              |          64.6993          |
    # |             310              |          66.8582          |
    # |             320              |          70.9373          |
    # |             330              |          70.7708          |
    # |             340              |          70.2931          |
    # |             350              |          62.3625          |
    # |             360              |          61.4102          |
    # |             370              |          68.3281          |
    # |             380              |          65.3496          |
    # |             390              |           66.217          |
    # |             400              |          66.5711          |


def s2c_cat2dog():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'vsi', 'total_variation']
    # metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    exp = Experiment(dataset="night2day", model="encomv3", name="encomv3_7.13.18.24_ttur_warmup_ep40_hid10",
                     load_size=256,
                     input_nc=3, output_nc=3, dataroot="/home/cas/home_ez/Datasets/EnCo", netG='munitv2', gpu_ids='0',
                     extra=' --no_dropout False')
    # exp.train(batch_size=1, n_epochs=20, n_epochs_decay=20, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_nls --stop_gradient True'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 5 --gan_mode lsgan')
    # exp.test(extra=' --batch_size 8', eval=False)
    # exp.fid()  # FID:
    # exp.traverse_fid()

    exp = Experiment(dataset="cat2wild", model="encomv3", name="encomv3_7.13.18.24_ttur_warmup_ep200_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot="/home/cas/home_ez/Datasets/EnCo", netG='munitv2',
                     )  # extra=' --no_dropout False'
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_nls --stop_gradient True'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 10')
    # exp.test(extra=' --batch_size 2', eval=False)
    # exp.fid()  # FID:
    # exp.traverse_fid()

    exp = Experiment(dataset="cityscapes", model="encov3", name="encov3_3,7,13,18,24,28_LN_ep200_hid10", load_size=256,
                     gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot="/home/cas/home_ez/Datasets/EnCo", extra=" --direction BtoA")
    # exp.swd(epoch='340')  # SWD: 21.0088
    exp = Experiment(dataset="cat2dog", model="encov33", name="encov33_7.13.18.24_ttur_warmup_ep200_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot="/home/cas/home_ez/Datasets/EnCo")
    # exp.swd(epoch='latest')  # SWD: 11.5737
    # exp.swd(epoch='180')  # SWD: 12.6669
    exp = Experiment(dataset="cat2dog", model="encov33", name="encov33_7.13.18.24_noDAG_ttur_warmup_ep200_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot="/home/cas/home_ez/Datasets/EnCo")
    # exp.swd(epoch='latest')  # SWD: 10.8346
    exp = Experiment(dataset="cityscapes", model="encov33", name="encov33_3,7,13,17,20,24_LN_ep200_id1", load_size=256,
                     gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot="/home/cas/home_ez/Datasets/EnCo", extra=" --direction BtoA")
    # exp.swd(epoch='340')
    # exp.swd(epoch='360')  # SWD: 23.5319

    exp = Experiment(dataset="cat2dog", model="encov33", name="encov33_7.13.18.24_noDAG_ttur_warmup_ep200_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot="/home/cas/home_ez/Datasets/EnCo")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,17,20,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_nls_v2 --stop_gradient True'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 10'
    #                 ' --oversample_ratio 1 --random_ratio 1')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:  54.15556198749877
    # exp.traverse_fid()
    #

    exp = Experiment(dataset="cat2dog", model="encov3", name="encov3_7.13.18.24_hinge_None_ep200_hid10", load_size=256,
                     input_nc=3, output_nc=3, dataroot="/home/yht/Casit/Datasets/ez/datasets", gpu_ids='0', )
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_nls --stop_gradient True'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 10 --prj_norm None')
    # exp.test(extra=' --batch_size 8', epoch='130')
    # exp.fid(epoch='130')  #
    # exp.traverse_fid()

    exp = Experiment(dataset="cat2dog", model="encov2", name="encov2_7.13.18.24_hinge_None_ep200_hid10", load_size=256,
                     input_nc=3, output_nc=3, dataroot="/home/yht/Casit/Datasets/ez/datasets", gpu_ids='0', )
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_nls --stop_gradient True'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 10 --prj_norm None')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  #
    # exp.traverse_fid()

    exp = Experiment(dataset="cat2dog", model="encov5", name="encov5_7.13.18.24_lsgan_ep200_hid10",
                     load_size=256, input_nc=3, output_nc=3, dataroot="/home/yht/Casit/Datasets/ez/datasets",
                     gpu_ids='0')
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_nls --stop_gradient True'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 10 --gan_mode lsgan')
    # exp.test(extra=' --batch_size 8', epoch='190')
    # exp.fid(epoch='190')  # FID:  59.001708798247876
    # exp.traverse_fid()

    exp = Experiment(dataset="cat2dog", model="encov3", name="encov3_7.13.18.24_ep200_hid1",
                     load_size=256, input_nc=3, output_nc=3, dataroot="/home/yht/Casit/Datasets/ez/datasets",
                     gpu_ids='0')
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 1 --lambda_NCE 10 --netF cam_mlp_sample_nl --stop_gradient True'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 10 --gan_mode lsgan')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:  67.8756, BAD!!!
    # exp.traverse_fid()

    exp = Experiment(dataset="horse2zebra_c", model="encov3", name="encov3_0,3,7,13,18,24,28,31_ttur_warmup_ep400_id1",
                     load_size=286, input_nc=3, output_nc=3, dataroot="/home/yht/Casit/Datasets/ez/datasets",
                     gpu_ids='0')
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 0,3,7,13,18,24,28,31 --display_freq 100'
    #                 ' --lambda_IDT 1 --lambda_NCE 10 --netF cam_mlp_sample_nl --stop_gradient True --num_patches 512'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 10 --stop_idt_epochs 400 --gan_mode lsgan')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:  118.1275130682701
    # exp.traverse_fid()
    # +------------------------------+------------------------+
    # |          epoch_num           |          FID           |
    # +------------------------------+------------------------+
    # |              10              |        94.4555         |
    # |              20              |        91.0343         |
    # |              30              |        49.5497         |
    # |              40              |        47.3508         |
    # |              50              |        50.4364         |
    # |              60              |        41.7147         |
    # |              70              |        44.8029         |
    # |              80              |        46.9903         |
    # |              90              |         76.489         |
    # |             100              |         54.719         |
    # |             110              |        39.4535         |
    # |             120              |        43.1148         |
    # |             130              |        35.6581         |
    # |             140              |        42.8453         |
    # |             150              |        49.9999         |
    # |             160              |        46.9547         |
    # |             170              |        46.1408         |
    # |             180              |        39.8228         |
    # |             190              |        38.9856         |
    # |             200              |        49.7249         |
    # |             210              |        36.5853         |
    # |             220              |        35.0642         |
    # |             230              |        36.6505         |
    # |             240              |        37.8246         |
    # |             250              |        35.5528         |
    # |             260              |        37.2058         |
    # |             270              |        36.8938         |
    # |             280              |        35.7161         |
    # |             290              |        38.2484         |
    # |             300              |        35.0648         |
    # |             310              |        34.9447         |
    # |             320              |        36.4098         |
    # |             330              |        36.5789         |
    # |             340              |        35.8917         |
    # |             350              |        35.0014         |
    # |             360              |        36.5739         |
    # |             370              |        36.8184         |
    # |             380              |         36.921         |
    # |             390              |        37.4705         |
    # |             400              |         37.405         |
    # +------------------------------+------------------------+

    exp = Experiment(dataset="horse2zebra_c", model="encov3", name="encov3_3,7,13,18,24,28_ttur_warmup_ep200_hid10",
                     load_size=256, input_nc=3, output_nc=3, dataroot="/home/yht/Casit/Datasets/ez/datasets",
                     gpu_ids='0')
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
    #                 ' --lambda_IDT 1 --lambda_NCE 10 --netF cam_mlp_sample_nls --stop_gradient True --num_patches 512'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 10 --stop_idt_epochs 400 --gan_mode lsgan')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:  118.1275130682701
    # exp.traverse_fid()
    # +----------------------------+-------------------------+
    # |         epoch_num          |           FID           |
    # +----------------------------+-------------------------+
    # |             10             |         229.285         |
    # |             20             |         142.9499        |
    # |             30             |         86.3289         |
    # |             40             |         66.9412         |
    # |             50             |         48.3624         |
    # |             60             |         47.8841         |
    # |             70             |         51.6087         |
    # |             80             |         40.8212         |
    # |             90             |         45.3693         |
    # |            100             |         64.7082         |
    # |            110             |         40.3199         |
    # |            120             |         41.8529         |
    # |            130             |         39.4343         |
    # |            140             |         104.9331        |
    # |            150             |         58.6299         |
    # |            160             |         42.3307         |
    # |            170             |         44.0841         |
    # |            180             |         40.6258         |
    # |            190             |         38.0979         |
    # |            200             |          69.117         |
    # |            210             |         33.9423         |
    # |            220             |         36.4642         |
    # |            230             |          35.558         |
    # |            240             |         39.5343         |
    # |            250             |         34.6485         |
    # |            260             |         38.9253         |
    # |            270             |         36.4964         |
    # |            280             |          34.92          |
    # |            290             |         35.8253         |
    # |            300             |         35.5457         |
    # |            310             |         33.9318         |
    # |            320             |         34.8703         |
    # |            330             |         36.2984         |
    # |            340             |         36.9175         |
    # |            350             |         36.8074         |
    # |            360             |         37.2535         |
    # |            370             |         35.6649         |
    # |            380             |         36.8205         |
    # |            390             |         36.6898         |
    # |            400             |         36.5993         |
    # +----------------------------+-------------------------+

    exp = Experiment(dataset="cat2dog", model="encov3", name="encov3_7.13.18.24_lsgan_ttur_warmup_ep200_hid10",
                     load_size=256, input_nc=3, output_nc=3, dataroot="/home/yht/Casit/Datasets/ez/datasets",
                     gpu_ids='0')
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_nls --stop_gradient True'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 10 --gan_mode lsgan')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:  118.1275130682701
    # exp.traverse_fid()
    # +--------------------------+-----------------------+
    # |        epoch_num         |          FID          |
    # +--------------------------+-----------------------+
    # |            10            |        162.035        |
    # |            20            |        132.8963       |
    # |            30            |        113.7876       |
    # |            40            |        103.1779       |
    # |            50            |        93.3035        |
    # |            60            |        88.8873        |
    # |            70            |        94.5326        |
    # |            80            |        80.9407        |
    # |            90            |        79.5545        |
    # |           100            |        71.7516        |
    # |           110            |        81.5911        |
    # |           120            |        78.9277        |
    # |           130            |        72.7522        |
    # |           140            |        64.0439        |
    # |           150            |        60.8191        |
    # |           160            |        62.6412        |
    # |           170            |        55.8789        |
    # |           180            |         56.869        |
    # |           190            |        56.3947        |
    # |           200            |        55.9622        |
    # +--------------------------+-----------------------+

    exp = Experiment(dataset="horse2zebra", model="enco", name="enco_lsa_3,7,13,18,24,28_ttur_warmup_ep400_hid5",
                     load_size=286, gpu_ids='0', input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False, netD='sa',
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
    #                 ' --stop_idt_epochs 200'
    #                 ' --lambda_IDT 5 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient True --gan_mode lsgan'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --flip_equivariance True'
    #                 ' --num_patches 256')
    # exp.test(extra=' --batch_size 8', epoch='latest')
    # exp.fid(epoch='latest')  # FID:
    # exp.traverse_fid()
    # +---------------------------+------------------------+
    # |         epoch_num         |          FID           |
    # +---------------------------+------------------------+
    # |             10            |        228.2912        |
    # |             20            |        139.9526        |
    # |             30            |         56.267         |
    # |             40            |        41.6513         |
    # |             50            |        56.1686         |
    # |             60            |        71.8091         |
    # |             70            |        71.9519         |
    # |             80            |        56.6982         |
    # |             90            |        76.6753         |
    # |            100            |        103.7112        |
    # |            110            |         49.79          |
    # |            120            |        119.8461        |
    # |            130            |        61.0057         |
    # |            140            |        77.0196         |
    # |            150            |        62.2808         |
    # |            160            |        61.8818         |
    # |            170            |         68.35          |
    # |            180            |        63.6023         |
    # |            190            |         90.368         |
    # |            200            |        66.1793         |
    # |            210            |        44.0241         |
    # |            220            |        51.4546         |
    # |            230            |         45.555         |
    # |            240            |        46.5634         |
    # |            250            |        41.3408         |
    # |            260            |        45.7869         |
    # |            270            |        46.1886         |
    # |            280            |        48.8652         |
    # |            290            |        43.0652         |
    # |            300            |        43.1771         |
    # |            310            |        44.8782         |
    # |            320            |        42.5569         |
    # |            330            |        45.5207         |
    # |            340            |         45.217         |
    # |            350            |        46.1879         |
    # |            360            |        43.7953         |
    # |            370            |        45.1605         |
    # |            380            |        45.0594         |
    # |            390            |        46.2472         |
    # |            400            |        45.1067         |
    # +---------------------------+------------------------+

    exp = Experiment(dataset="horse2zebra", model="enco", name="enco_lsa_3,7,13,18,24,28_ttur_warmup_ep400_id5",
                     load_size=286, gpu_ids='0', input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False, netD='sa',
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
    #                 ' --stop_idt_epochs 400'
    #                 ' --lambda_IDT 5 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient True --gan_mode lsgan'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --flip_equivariance True'
    #                 ' --num_patches 256')
    # exp.test(extra=' --batch_size 8', epoch='latest')
    # exp.fid(epoch='latest')  # FID:
    # exp.traverse_fid()
    # # +---------------------------+------------------------+
    # |             10            |        188.8907        |
    # |             20            |        111.7686        |
    # |             30            |        54.0254         |
    # |             40            |        55.4346         |
    # |             50            |         53.825         |
    # |             60            |        68.9259         |
    # |             70            |        50.0533         |
    # |             80            |        93.5283         |
    # |             90            |        51.9927         |
    # |            100            |        62.9474         |
    # |            110            |        60.4145         |
    # |            120            |         62.757         |
    # |            130            |        50.8169         |
    # |            140            |         56.505         |
    # |            150            |        66.2019         |
    # |            160            |        57.0728         |
    # |            170            |        57.4059         |
    # |            180            |        66.2168         |
    # |            190            |        53.7749         |
    # |            200            |        41.7683         |
    # |            210            |         36.477         |
    # |            220            |        41.3774         |
    # |            230            |        79.6147         |
    # |            240            |        47.0802         |
    # |            250            |         43.733         |
    # |            260            |        41.0758         |
    # |            270            |        46.2636         |
    # |            280            |         41.654         |
    # |            290            |        44.7225         |
    # |            300            |        38.5332         |
    # |            310            |        40.3072         |
    # |            320            |        38.3791         |
    # |            330            |        39.4027         |
    # |            340            |        42.0775         |
    # |            350            |        42.0979         |
    # |            360            |        39.2718         |
    # |            370            |        40.2149         |
    # |            380            |        39.4109         |
    # |            390            |        39.0877         |
    # |            400            |        39.4662         |
    # +---------------------------+------------------------+

    exp = Experiment(dataset="horse2zebra", model="enco", name="enco_lsa_3,7,13,18,24,28_ttur_warmup_ep400_id10",
                     load_size=286, gpu_ids='0', input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False, netD='sa',
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
    #                 ' --stop_idt_epochs 200'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient True --gan_mode lsgan'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --flip_equivariance True'
    #                 ' --num_patches 128')
    # exp.test(extra=' --batch_size 8', epoch='latest')
    # exp.fid(epoch='latest')  # FID:
    # exp.traverse_fid()
    # +---------------------------+------------------------+
    # |         epoch_num         |          FID           |
    # +---------------------------+------------------------+
    # |             10            |        123.4146        |
    # |             20            |         86.346         |
    # |             30            |        54.6899         |
    # |             40            |        57.5301         |
    # |             50            |        47.4252         |
    # |             60            |        46.8172         |
    # |             70            |        60.7969         |
    # |             80            |        44.8971         |
    # |             90            |        55.9524         |
    # |            100            |        83.9948         |
    # |            110            |        73.7814         |
    # |            120            |        60.7681         |
    # |            130            |        162.8601        |
    # |            140            |         55.927         |
    # |            150            |        89.0718         |
    # |            160            |        66.0036         |
    # |            170            |        53.1065         |
    # |            180            |        67.7256         |
    # |            190            |        46.0406         |
    # |            200            |        56.8989         |
    # |            210            |        47.0175         |
    # |            220            |        54.4379         |
    # |            230            |        45.5113         |
    # |            240            |        38.8554         |
    # |            250            |        38.9452         |
    # |            260            |        40.8658         |
    # |            270            |        39.4729         |
    # |            280            |        36.6119         |
    # |            290            |        40.8268         |
    # |            300            |         39.551         |
    # |            310            |         42.958         |
    # |            320            |        42.8389         |
    # |            330            |        40.7655         |
    # |            340            |        40.0054         |
    # |            350            |        40.5463         |
    # |            360            |        40.8317         |
    # |            370            |        41.4198         |
    # |            380            |         43.086         |
    # |            390            |        42.1856         |
    # |            400            |        42.2673         |
    # +---------------------------+------------------------+

    exp = Experiment(dataset="horse2zebra", model="enco", name="enco_lsa_3,7,13,18,24,28_ttur_warmup_ep400_hid10",
                     load_size=286, gpu_ids='0', input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False, netD='sa',
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
    #                 ' --stop_idt_epochs 400'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient True --gan_mode lsgan'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --flip_equivariance True'
    #                 ' --num_patches 128')
    # exp.test(extra=' --batch_size 8', epoch='latest')
    # exp.fid(epoch='latest')  # FID:
    # exp.traverse_fid()
    # +----------------------------+-------------------------+
    # |         epoch_num          |           FID           |
    # +----------------------------+-------------------------+
    # |             10             |         149.6069        |
    # |             20             |         102.7962        |
    # |             30             |         111.1705        |
    # |             40             |         53.8825         |
    # |             50             |         74.1638         |
    # |             60             |         99.1604         |
    # |             70             |          53.955         |
    # |             80             |         60.4426         |
    # |             90             |         58.2374         |
    # |            100             |         54.1734         |
    # |            110             |          54.132         |
    # |            120             |         48.9943         |
    # |            130             |         50.4661         |
    # |            140             |         45.4816         |
    # |            150             |         75.3109         |
    # |            160             |         54.1697         |
    # |            170             |         51.3974         |
    # |            180             |         44.7623         |
    # |            190             |         53.4625         |
    # |            200             |         37.8555         |
    # |            210             |         47.6868         |
    # |            220             |          64.042         |
    # |            230             |         44.0341         |
    # |            240             |         39.9873         |
    # |            250             |         48.9742         |
    # |            260             |         53.4491         |
    # |            270             |          35.255         |
    # |            280             |         41.7862         |
    # |            290             |         37.3888         |
    # |            300             |          38.848         |
    # |            310             |         40.0374         |
    # |            320             |          38.861         |
    # |            330             |          41.586         |
    # |            340             |         43.1853         |
    # |            350             |         43.0664         |
    # |            360             |         41.8997         |
    # |            370             |         42.4426         |
    # |            380             |          41.959         |
    # |            390             |         44.1234         |
    # |            400             |         43.4769         |
    # +----------------------------+-------------------------+

    exp = Experiment(dataset="horse2zebra", model="enco", name="enco_lsa_3,7,13,18,24,28_ttur_warmup_ep400_id1",
                     load_size=286, gpu_ids='0', input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False, netD='sa',
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
    #                 ' --stop_idt_epochs 400'
    #                 ' --lambda_IDT 1 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient True --gan_mode lsgan'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --flip_equivariance True'
    #                 ' --num_patches 128')
    # exp.test(extra=' --batch_size 8', epoch='latest')
    # exp.fid(epoch='latest')  # FID:
    # exp.traverse_fid()
    # +---------------------------+------------------------+
    # |         epoch_num         |          FID           |
    # +---------------------------+------------------------+
    # |             10            |        226.0674        |
    # |             20            |        140.2277        |
    # |             30            |        66.3954         |
    # |             40            |        121.5277        |
    # |             50            |        97.3125         |
    # |             60            |         58.108         |
    # |             70            |         48.479         |
    # |             80            |        35.4979         |
    # |             90            |        48.8008         |
    # |            100            |         60.231         |
    # |            110            |        43.5612         |
    # |            120            |        59.8505         |
    # |            130            |        63.1856         |
    # |            140            |        38.9269         |
    # |            150            |        87.0509         |
    # |            160            |        53.1873         |
    # |            170            |        66.3888         |
    # |            180            |        66.3275         |
    # |            190            |        96.4829         |
    # |            200            |        90.2112         |
    # |            210            |        43.0824         |
    # |            220            |        47.9534         |
    # |            230            |        41.1489         |
    # |            240            |        43.0421         |
    # |            250            |        47.3326         |
    # |            260            |        38.1382         |
    # |            270            |        50.0701         |
    # |            280            |        39.7448         |
    # |            290            |        38.5708         |
    # |            300            |        43.0334         |
    # |            310            |        44.4629         |
    # |            320            |        43.2246         |
    # |            330            |        39.8921         |
    # |            340            |        45.2298         |
    # |            350            |        45.5091         |
    # |            360            |        44.2737         |
    # |            370            |        41.1061         |
    # |            380            |        44.4838         |
    # |            390            |         42.352         |
    # |            400            |        43.0953         |
    # +---------------------------+------------------------+

    exp = Experiment(dataset="horse2zebra", model="cutcamv4",
                     name="cutcamv4_128_3.7.13.18.24.28_ttur_warmup_ep400_id10",
                     load_size=286, gpu_ids='0', input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100 --stop_idt_epochs 400'
    #                 ' --lambda_IDT 1 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient False --gan_mode lsgan'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --flip_equivariance True'
    #                 ' --num_patches 128')
    # exp.test(extra=' --batch_size 8', epoch='latest')
    # exp.fid(epoch='latest')  # FID:
    # exp.traverse_fid()
    # +-----------------------------+--------------------------+
    # |          epoch_num          |           FID            |
    # +-----------------------------+--------------------------+
    # |              10             |         223.755          |
    # |              20             |         103.0487         |
    # |              30             |         80.4346          |
    # |              40             |         123.7414         |
    # |              50             |         203.468          |
    # |              60             |         45.8066          |
    # |              70             |          45.988          |
    # |              80             |         53.0925          |
    # |              90             |         45.7831          |
    # |             100             |         56.7068          |
    # |             110             |         72.1454          |
    # |             120             |          53.077          |
    # |             130             |         56.1354          |
    # |             140             |         48.7379          |
    # |             150             |         66.2563          |
    # |             160             |         54.4703          |
    # |             170             |         42.1473          |
    # |             180             |         39.8913          |
    # |             190             |         61.6496          |
    # |             200             |         40.1978          |
    # |             210             |         45.2901          |
    # |             220             |         52.3755          |
    # |             230             |         48.2581          |
    # |             240             |          67.262          |
    # |             250             |         47.9947          |
    # |             260             |         48.4709          |
    # |             270             |         51.0099          |
    # |             280             |         44.2871          |
    # |             290             |         42.5839          |
    # |             300             |         43.6274          |
    # |             310             |         42.3106          |
    # |             320             |         44.3104          |
    # |             330             |         43.8701          |
    # |             340             |         41.1987          |
    # |             350             |         41.6354          |
    # |             360             |         41.9804          |
    # |             370             |         42.0498          |
    # |             380             |         42.7222          |
    # |             390             |         43.3126          |
    # |             400             |         43.2352          |
    # +-----------------------------+--------------------------+

    exp = Experiment(dataset="horse2zebra", model="enco", name="enco_3.7.13.18.24.28_ttur_warmup_ep400_hid10",
                     load_size=256, gpu_ids='0', input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=True,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
    #                 ' --stop_idt_epochs 200'
    #                 ' --lambda_IDT 1 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient False --gan_mode lsgan'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 100 --flip_equivariance True')
    # exp.test(extra=' --batch_size 8', epoch='latest')
    # exp.fid(epoch='latest')  # FID:
    # exp.traverse_fid()
    # +--------------------------+-----------------------+
    # |        epoch_num         |          FID          |
    # +--------------------------+-----------------------+
    # |            10            |        206.9148       |
    # |            20            |        151.6908       |
    # |            30            |         70.801        |
    # |            40            |        77.1846        |
    # |            50            |        61.2902        |
    # |            60            |        69.9005        |
    # |            70            |        62.7851        |
    # |            80            |        161.6081       |
    # |            90            |        65.8053        |
    # |           100            |        67.3428        |
    # |           110            |        42.1372        |
    # |           120            |        68.1658        |
    # |           130            |        57.5457        |
    # |           140            |        50.6273        |
    # |           150            |        69.5493        |
    # |           160            |        53.1886        |
    # |           170            |        50.7016        |
    # |           180            |        56.6155        |
    # |           190            |        47.4478        |
    # |           200            |         64.788        |
    # |           210            |        50.1171        |
    # |           220            |        43.2063        |
    # |           230            |        44.7789        |
    # |           240            |        45.6501        |
    # |           250            |        50.0415        |
    # |           260            |        41.2432        |
    # |           270            |        45.8565        |
    # |           280            |        41.5347        |
    # |           290            |        42.6764        |
    # |           300            |        39.6665        |
    # |           310            |        42.1239        |
    # |           320            |        43.3379        |
    # |           330            |        42.8209        |
    # |           340            |        45.6637        |
    # |           350            |        45.0806        |
    # |           360            |        44.8161        |
    # |           370            |        44.5422        |
    # |           380            |        45.7601        |
    # |           390            |        45.4462        |
    # |           400            |        45.1811        |
    # +--------------------------+-----------------------+

    exp = Experiment(dataset="horse2zebra", model="enco", name="enco_twoF_lsgan_3.7.13.18.24.28_ttur_warmup_ep400_id1",
                     load_size=286, gpu_ids='0', input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100 --stop_idt_epochs 400'
    #                 ' --lambda_IDT 1 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient false --gan_mode lsgan'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --flip_equivariance true'
    #                 ' --two_F True')
    #
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # exp.test(script='eval_piq')
    # +-----------------------------+--------------------------+
    # |          epoch_num          |           FID            |
    # +-----------------------------+--------------------------+
    # |              10             |         186.6848         |
    # |              20             |         184.949          |
    # |              30             |          64.413          |
    # |              40             |         53.0178          |
    # |              50             |         56.4882          |
    # |              60             |         59.4746          |
    # |              70             |          45.57           |
    # |              80             |         60.1728          |
    # |              90             |         48.5971          |
    # |             100             |         69.9432          |
    # |             110             |         52.7503          |
    # |             120             |         141.662          |
    # |             130             |         51.6494          |
    # |             140             |         54.2809          |
    # |             150             |         76.5074          |
    # |             160             |         52.5825          |
    # |             170             |         46.0562          |
    # |             180             |         47.3765          |
    # |             190             |         59.5966          |
    # |             200             |         51.9095          |
    # |             210             |         40.4072          |
    # |             220             |         56.5581          |
    # |             230             |         57.6085          |
    # |             240             |          45.311          |
    # |             250             |         39.7351          |
    # |             260             |         58.8668          |
    # |             270             |         41.5864          |
    # |             280             |         49.3832          |
    # |             290             |         50.1271          |
    # |             300             |         47.3587          |
    # |             310             |         47.7529          |
    # |             320             |          45.321          |
    # |             330             |         45.6074          |
    # |             340             |         48.0465          |
    # |             350             |         47.3665          |
    # |             360             |         47.4447          |
    # |             370             |         46.1976          |
    # |             380             |         45.8037          |
    # |             390             |         45.0845          |
    # |             400             |         46.8583          |
    # +-----------------------------+--------------------------+

    exp = Experiment(dataset="horse2zebra", model="cutcamv4",
                     name="cutcamv4_lsgan_4,8,13,18,21,25_ttur_warmup_ep400_id1",
                     load_size=286, gpu_ids='0', input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 4,8,13,18,21,25 --display_freq 100 --stop_idt_epochs 400'
    #                 ' --lambda_IDT 1 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient false --gan_mode lsgan'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --flip_equivariance true')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID: bad
    # exp.traverse_fid()

    exp = Experiment(dataset="horse2zebra", model="cutcamv4",
                     name="cutcamv4_lsgan_3.7.13.18.24.28_ttur_warmup_ep400_id10",
                     load_size=286, gpu_ids='0', input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=True,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100 --stop_idt_epochs 400'
    #                 ' --lambda_IDT 1 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient False --gan_mode lsgan'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --flip_equivariance True')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +-------------------------------+------------------------+
    # |           epoch_num           |          FID           |
    # +-------------------------------+------------------------+
    # |               10              |         57.282         |
    # |               20              |        50.0264         |
    # |               30              |        42.1942         |
    # |               40              |        42.5149         |
    # |               50              |        65.1589         |
    # |               60              |         47.286         |
    # |               70              |        46.5418         |
    # |               80              |        54.1184         |
    # |               90              |        41.1212         |
    # |              100              |        50.4032         |
    # |              110              |        48.7928         |
    # |              120              |        57.3205         |
    # |              130              |         66.333         |
    # |              140              |        46.0152         |
    # |              150              |        45.2135         |
    # |              160              |        39.3884         |
    # |              170              |        35.5647         |
    # |              180              |        50.3957         |
    # |              190              |        41.9456         |
    # |              200              |        42.2288         |
    # |              210              |        39.4061         |
    # |              220              |        42.1263         |
    # |              230              |        38.2239         |
    # |              240              |        38.9548         |
    # |              250              |        46.7531         |
    # |              260              |        41.0801         |
    # |              270              |        46.9665         |
    # |              280              |        44.8889         |
    # |              290              |        47.9157         |
    # |              300              |        44.9377         |
    # |              310              |        41.9713         |
    # |              320              |        45.6528         |
    # |              330              |        45.5238         |
    # |              340              |        43.2464         |
    # |              350              |        45.0417         |
    # |              360              |        42.6628         |
    # |              370              |        45.2103         |
    # |              380              |        46.6821         |
    # |              390              |        45.4857         |
    # |              400              |         45.923         |
    # +-------------------------------+------------------------+

    exp = Experiment(dataset="horse2zebra", model="cutcamv4",
                     name="cutcamv4_lsgan_0.3.7.13.18.24.28.31_ttur_warmup_ep400_id10",
                     load_size=286, gpu_ids='0', input_nc=3, output_nc=3, dataroot=r"../datasets", )
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 0,3,7,13,18,24,28,31 --display_freq 100 --stop_idt_epochs 400'
    #                 ' --lambda_IDT 1 --lambda_NCE 1 --netF cam_mlp_sample_s --stop_gradient False --gan_mode lsgan'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --flip_equivariance True')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +-------------------------------+----------------------------+
    # |           epoch_num           |            FID             |
    # +-------------------------------+----------------------------+
    # |               10              |          136.2567          |
    # |               20              |          124.1591          |
    # |               30              |           76.934           |
    # |               40              |          75.7383           |
    # |               50              |          49.3793           |
    # |               60              |          78.1652           |
    # |               70              |          76.0351           |
    # |               80              |          55.6905           |
    # |               90              |          48.6658           |
    # |              100              |          88.5243           |
    # |              110              |          43.9754           |
    # |              120              |          49.4065           |
    # |              130              |           52.513           |
    # |              140              |          72.5203           |
    # |              150              |          50.6907           |
    # |              160              |          53.9601           |
    # |              170              |          79.1023           |
    # |              180              |          36.8099           |
    # |              190              |          65.3973           |
    # |              200              |           51.823           |
    # |              210              |          49.1626           |
    # |              220              |          71.4984           |
    # |              230              |          67.2164           |
    # |              240              |           51.253           |
    # |              250              |          47.7384           |
    # |              260              |           56.214           |
    # |              270              |          51.3249           |
    # |              280              |          41.5254           |
    # |              290              |          46.8907           |
    # |              300              |          49.0325           |
    # |              310              |          46.1604           |
    # |              320              |          48.5054           |
    # |              330              |          50.2342           |
    # |              340              |          48.3916           |
    # |              350              |           50.996           |
    # |              360              |          49.9754           |
    # |              370              |           51.852           |
    # |              380              |          52.7827           |
    # |              390              |          53.4101           |
    # |              400              |          53.5486           |
    # +-------------------------------+----------------------------+

    exp = Experiment(dataset="horse2zebra", model="cutcamv4",
                     name="cutcamv4_lsgan_0.3.7.13.18.24.28.31_ttur_warmup_ep400_id10",
                     load_size=286, gpu_ids='0', input_nc=3, output_nc=3, dataroot=r"../datasets", )
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 0,3,7,13,18,24,28,31 --display_freq 100 --stop_idt_epochs 400'
    #                 ' --lambda_IDT 1 --lambda_NCE 1 --netF cam_mlp_sample_s --stop_gradient False --gan_mode lsgan'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --flip_equivariance True')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()

    exp = Experiment(dataset="horse2zebra", model="cutcamv4",
                     name="BtoA_cutcamv4_lsgan_0.3.7.13.18.24.28.31_ttur_warmup_ep400_id10",
                     load_size=286, gpu_ids='0', input_nc=3, output_nc=3, dataroot=r"../datasets",
                     extra=' --direction BtoA')
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 0,3,7,13,18,24,28,31 --display_freq 100 --stop_idt_epochs 400'
    #                 ' --lambda_IDT 1 --lambda_NCE 1 --netF cam_mlp_sample_s --stop_gradient False --gan_mode lsgan'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --flip_equivariance True')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +---------------------------------+------------------------------+
    # |            epoch_num            |             FID              |
    # +---------------------------------+------------------------------+
    # |                10               |           245.3767           |
    # |                20               |           197.2421           |
    # |                30               |           189.6982           |
    # |                40               |           201.1023           |
    # |                50               |           179.5353           |
    # |                60               |           197.4652           |
    # |                70               |           194.772            |
    # |                80               |           163.9331           |
    # |                90               |           177.4471           |
    # |               100               |           149.7752           |
    # |               110               |           161.1771           |
    # |               120               |           145.8596           |
    # |               130               |           160.6334           |
    # |               140               |           149.7242           |
    # |               150               |           152.6791           |
    # |               160               |           161.6244           |
    # |               170               |           156.8414           |
    # |               180               |           172.3587           |
    # |               190               |           151.4034           |
    # |               200               |           148.6369           |
    # |               210               |           150.8584           |
    # |               220               |           160.2274           |
    # |               230               |           155.5571           |
    # |               240               |           154.2063           |
    # |               250               |           147.2445           |
    # |               260               |           146.1838           |
    # |               270               |           155.5383           |
    # |               280               |           145.7174           |
    # |               290               |           154.3773           |
    # |               300               |           153.4855           |
    # |               310               |           150.6617           |
    # |               320               |           150.1282           |
    # |               330               |           146.101            |
    # |               340               |           152.5113           |
    # |               350               |           152.2595           |
    # |               360               |           150.716            |
    # |               370               |           152.4948           |
    # |               380               |           151.8324           |
    # |               390               |           151.6147           |
    # |               400               |           152.812            |
    # +---------------------------------+------------------------------+

    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_lsgan_7.13.18.24_ttur_warmup_ep200_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient False'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 10 --gan_mode lsgan')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +--------------------------+------------------------+
    # |        epoch_num         |          FID           |
    # +--------------------------+------------------------+
    # |            10            |        139.9316        |
    # |            20            |        116.9679        |
    # |            30            |        120.0194        |
    # |            40            |        118.8593        |
    # |            50            |        95.7766         |
    # |            60            |        93.0505         |
    # |            70            |        87.4753         |
    # |            80            |        85.2426         |
    # |            90            |        77.6028         |
    # |           100            |        71.8168         |
    # |           110            |        94.8929         |
    # |           120            |        103.547         |
    # |           130            |        73.8297         |
    # |           140            |        73.4081         |
    # |           150            |        64.0776         |
    # |           160            |        64.3899         |
    # |           170            |         58.646         |
    # |           180            |        60.3845         |
    # |           190            |        57.2116         |
    # |           200            |        57.0297         |
    # +--------------------------+------------------------+

    exp = Experiment(dataset="horse2zebra", model="cutcamv4",
                     name="BtoA_cutcamv4_lsgan_3.7.13.18.24.28_ttur_warmup_ep400_id10",
                     load_size=286, gpu_ids='0', input_nc=3, output_nc=3, dataroot=r"../datasets",
                     extra=' --direction BtoA')
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100 --stop_idt_epochs 400'
    #                 ' --lambda_IDT 1 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient False --gan_mode lsgan'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --flip_equivariance True')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +-------------------------------+----------------------------+
    # |           epoch_num           |            FID             |
    # +-------------------------------+----------------------------+
    # |               10              |          219.8832          |
    # |               20              |          199.5127          |
    # |               30              |          213.2062          |
    # |               40              |          211.8756          |
    # |               50              |          194.0136          |
    # |               60              |          164.8692          |
    # |               70              |          193.8908          |
    # |               80              |          225.1205          |
    # |               90              |          205.4191          |
    # |              100              |          193.7186          |
    # |              110              |          176.6052          |
    # |              120              |          178.7192          |
    # |              130              |          168.9096          |
    # |              140              |          145.9409          |
    # |              150              |          167.3586          |
    # |              160              |          189.7128          |
    # |              170              |          231.5303          |
    # |              180              |          149.012           |
    # |              190              |          141.8816          |
    # |              200              |          189.1017          |
    # |              210              |          146.727           |
    # |              220              |          154.6789          |
    # |              230              |          151.1374          |
    # |              240              |          134.3968          |
    # |              250              |          177.4181          |
    # |              260              |          141.7941          |
    # |              270              |          158.5474          |
    # |              280              |          150.4454          |
    # |              290              |          142.9977          |
    # |              300              |          155.8905          |
    # |              310              |          149.0062          |
    # |              320              |          137.271           |
    # |              330              |          145.3731          |
    # |              340              |          140.5891          |
    # |              350              |          144.5717          |
    # |              360              |          145.3269          |
    # |              370              |          146.3232          |
    # |              380              |          145.0875          |
    # |              390              |          146.7959          |
    # |              400              |          145.8588          |
    # +-------------------------------+----------------------------+

    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_lsgan_7.13.18.24_ttur_warmup_ep200_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient False'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 10 --gan_mode lsgan')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:  118.1275130682701
    # exp.traverse_fid()

    exp = Experiment(dataset="vangogh2photo", model="cutcamv4", name="cutcamv4_sam_7.13.18.24_nosg_ttur_ep200_id1",
                     load_size=286, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets", extra=' --direction BtoA')
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 1 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient False --gan_mode lsgan'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --stop_idt_epochs 200')
    # exp.traverse_fid()
    #
    # exp.test(epoch='latest', extra=' --batch_size 8')
    # exp.fid(epoch='latest')  # FID:
    # metric_list = ['dists', 'lpips']
    # exp.eval(metric_list)

    # +--------------------------+------------------------+
    # |        epoch_num         |          FID           |
    # +--------------------------+------------------------+
    # |            10            |        160.0179        |
    # |            20            |        188.7069        |
    # |            30            |        128.3865        |
    # |            40            |        133.9157        |
    # |            50            |        130.963         |
    # |            60            |        150.9769        |
    # |            70            |        134.7732        |
    # |            80            |        129.5504        |
    # |            90            |        126.1994        |
    # |           100            |        125.2904        |
    # |           110            |        125.0714        |
    # |           120            |        128.4931        |
    # |           130            |        134.7752        |
    # |           140            |        126.9932        |
    # |           150            |        128.6083        |
    # |           160            |        125.5151        |
    # |           170            |        125.7166        |
    # |           180            |        126.1789        |
    # |           190            |        127.3237        |
    # |           200            |        127.2785        |
    # +--------------------------+------------------------+
    # +---------------------------------------------+--------+--------+
    # |                    method                   | dists  | lpips  |
    # +---------------------------------------------+--------+--------+
    # | cutcamv4_sam_7.13.18.24_nosg_ttur_ep200_id1 | 0.3558 | 0.7416 |
    # +---------------------------------------------+--------+--------+

    exp = Experiment(dataset="cityscapes", model="cutcamv4",
                     name="cutcamv4_lsgan_3.7.13.18.24.28_ttur_warmup_ep400_hid1", load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets", extra=' --direction BtoA')
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
    #                 ' --lambda_IDT 1 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient False'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --gan_mode lsgan'
    #                 ' --stop_idt_epochs 400')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # exp.test(extra=' --batch_size 8')
    # exp.eval_cityscapes()
    # [2022-07-30 00:10:53,238 segment.py:582 test_more] ===> mAP 22.930
    # [2022-07-30 00:10:53,238 segment.py:584 test_more] ===> pixAcc 68.458
    # [2022-07-30 00:10:53,239 segment.py:585 test_more] ===> classAcc 30.661
    # [2022-07-30 00:10:53,239 segment.py:586 test_more] ===> mIoU 22.927
    # +-----------------------------+--------------------------+
    # |          epoch_num          |           FID            |
    # +-----------------------------+--------------------------+
    # |              10             |         246.3282         |
    # |              20             |         204.9931         |
    # |              30             |         174.7112         |
    # |              40             |         188.711          |
    # |              50             |         115.8203         |
    # |              60             |         93.6053          |
    # |              70             |         82.8434          |
    # |              80             |         74.3597          |
    # |              90             |         85.5669          |
    # |             100             |         64.9839          |
    # |             110             |         67.9763          |
    # |             120             |         62.2351          |
    # |             130             |         58.7331          |
    # |             140             |         61.3628          |
    # |             150             |         56.3026          |
    # |             160             |         57.6486          |
    # |             170             |         61.0389          |
    # |             180             |         65.0958          |
    # |             190             |         60.9907          |
    # |             200             |         215.4069         |
    # |             210             |         62.7539          |
    # |             220             |         54.4422          |
    # |             230             |         56.6701          |
    # |             240             |         55.8226          |
    # |             250             |         57.5408          |
    # |             260             |         50.5655          |
    # |             270             |         55.8513          |
    # |             280             |         51.9279          |
    # |             290             |         52.1183          |
    # |             300             |         53.4721          |
    # |             310             |         54.0733          |
    # |             320             |          50.449          |
    # |             330             |         50.8779          |
    # |             340             |         51.6595          |
    # |             350             |         50.0443          |
    # |             360             |         49.0703          |
    # |             370             |         49.0856          |
    # |             380             |         48.4394          |
    # |             390             |         48.7605          |
    # |             400             |         48.5162          |
    # +-----------------------------+--------------------------+

    exp = Experiment(dataset="cityscapes", model="cutv4", name="cutv4_lsgan_3.7.13.18.24.28_ttur_warmup_ep400_id1",
                     load_size=256, gpu_ids='0', input_nc=3, output_nc=3, dataroot=r"../datasets",
                     extra=' --direction BtoA')
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
    #                 ' --lambda_IDT 1 --lambda_NCE 2 --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --gan_mode lsgan')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # exp.eval_cityscapes()

    exp = Experiment(dataset="cat2dog", model="cutcamv4",
                     name="cutcamv4_sam_nosg_3.7.13.18.24.28_ttur_warmup_ep200_hid10", load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient False'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 10 --gan_mode lsgan')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:  118.1275130682701
    # exp.traverse_fid()
    # +-----------------------------+--------------------------+
    # |          epoch_num          |           FID            |
    # +-----------------------------+--------------------------+
    # |              10             |         165.4053         |
    # |              20             |         130.3258         |
    # |              30             |         120.2709         |
    # |              40             |         113.1878         |
    # |              50             |         106.4976         |
    # |              60             |         97.1321          |
    # |              70             |          91.534          |
    # |              80             |         88.0054          |
    # |              90             |         94.9534          |
    # |             100             |         89.8881          |
    # |             110             |         96.1745          |
    # |             120             |         85.5665          |
    # |             130             |         77.6165          |
    # |             140             |         68.0066          |
    # |             150             |         74.5723          |
    # |             160             |          66.536          |
    # |             170             |         67.2215          |
    # |             180             |         62.2975          |
    # |             190             |         63.3343          |
    # |             200             |         62.2699          |
    # +-----------------------------+--------------------------+

    exp = Experiment(dataset="horse2zebra", model="cutcamv4",
                     name="BtoA_cutcamv4_ralsgan_nosg_7,13,18,24_ttur_warmup_ep400_id10", load_size=286, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets", extra=' --gpu_ids 0')
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 4 --netF cam_mlp_sample --stop_gradient False'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --gan_mode ralsgan')
    # exp.test(extra=' --batch_size 8', epoch='400')
    # exp.fid(epoch='400')  # FID: 119.90970059860655
    # exp.traverse_fid()

    exp = Experiment(dataset="horse2zebra", model="cutcamv4", name="cutcamv4_lsgan_7.13.18.24_ttur_warmup_ep400_id10",
                     load_size=286, gpu_ids='0', input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient False --gan_mode lsgan'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --flip_equivariance True')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +----------------------------+-------------------------+
    # |         epoch_num          |           FID           |
    # +----------------------------+-------------------------+
    # |             10             |         162.6761        |
    # |             20             |         157.2743        |
    # |             30             |         75.4545         |
    # |             40             |         123.6617        |
    # |             50             |         166.2632        |
    # |             60             |         110.2566        |
    # |             70             |         56.6553         |
    # |             80             |         82.7711         |
    # |             90             |         54.0764         |
    # |            100             |         50.7065         |
    # |            110             |         60.2318         |
    # |            120             |          53.268         |
    # |            130             |         56.6986         |
    # |            140             |         73.9111         |
    # |            150             |         51.2102         |
    # |            160             |         68.3881         |
    # |            170             |         76.4654         |
    # |            180             |         72.3432         |
    # |            190             |         59.3008         |
    # |            200             |         39.9135         |
    # |            210             |         58.6505         |
    # |            220             |         50.1698         |
    # |            230             |         43.7997         |
    # |            240             |          46.042         |
    # |            250             |         45.8256         |
    # |            260             |         51.0333         |
    # |            270             |         44.9911         |
    # |            280             |         50.5104         |
    # |            290             |         46.6757         |
    # |            300             |         47.7886         |
    # |            310             |         48.8526         |
    # |            320             |         48.8181         |
    # |            330             |         45.8229         |
    # |            340             |         50.2769         |
    # |            350             |          52.705         |
    # |            360             |         49.0316         |
    # |            370             |         47.5024         |
    # |            380             |         48.2676         |
    # |            390             |         48.0647         |
    # |            400             |         47.8357         |
    # +----------------------------+-------------------------+

    exp = Experiment(dataset="horse2zebra", model="cutcamv4", name="cutcamv4_nosg_7.13.18.24_ttur_warmup_ep400_id1",
                     load_size=286, gpu_ids='0', input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 5 --netF cam_mlp_sample_s --stop_gradient False --gan_mode lsgan'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +---------------------------+------------------------+
    # |         epoch_num         |          FID           |
    # +---------------------------+------------------------+
    # |             10            |        113.1116        |
    # |             20            |        133.3075        |
    # |             30            |        60.8226         |
    # |             40            |        93.1509         |
    # |             50            |        98.5243         |
    # |             60            |        55.4076         |
    # |             70            |        46.4619         |
    # |             80            |        76.9269         |
    # |             90            |        55.2502         |
    # |            100            |        56.0827         |
    # |            110            |        47.9786         |
    # |            120            |        79.3402         |
    # |            130            |        52.0303         |
    # |            140            |        63.5925         |
    # |            150            |        52.4412         |
    # |            160            |        64.7223         |
    # |            170            |        56.9998         |
    # |            180            |        52.9577         |
    # |            190            |        49.2642         |
    # |            200            |        47.6892         |
    # |            210            |        52.6584         |
    # |            220            |        54.2542         |
    # |            230            |        58.0235         |
    # |            240            |        45.0089         |
    # |            250            |        58.7093         |
    # |            260            |        55.0037         |
    # |            270            |        43.9144         |
    # |            280            |        43.3959         |
    # |            290            |        46.9985         |
    # |            300            |        44.8038         |
    # |            310            |        46.9346         |
    # |            320            |        45.1914         |
    # |            330            |        41.6554         |
    # |            340            |        44.9858         |
    # |            350            |        44.5467         |
    # |            360            |        44.1797         |
    # |            370            |        43.9395         |
    # |            380            |         43.346         |
    # |            390            |        43.6511         |
    # |            400            |        45.0431         |
    # +---------------------------+------------------------+

    exp = Experiment(dataset="horse2zebra", model="cutcamv4", name="cutcamv4_nosg_7.13.18.24_ttur_warmup_ep400_id5",
                     load_size=256, gpu_ids=0, input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=20, n_epochs_decay=380, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 5 --lambda_NCE 5 --netF cam_mlp_sample --stop_gradient False'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --stop_idt_epochs 400')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +---------------------------+------------------------+
    # |         epoch_num         |          FID           |
    # +---------------------------+------------------------+
    # |             10            |        334.5698        |
    # |             20            |        83.4809         |
    # |             30            |         42.871         |
    # |             40            |        71.2591         |
    # |             50            |        67.2129         |
    # |             60            |        56.6825         |
    # |             70            |        48.3458         |
    # |             80            |        45.0133         |
    # |             90            |        43.2974         |
    # |            100            |        50.3453         |
    # |            110            |         40.835         |
    # |            120            |        62.3308         |
    # |            130            |         39.869         |
    # |            140            |        39.6102         |
    # |            150            |        44.7164         |
    # |            160            |        50.0595         |
    # |            170            |        46.0336         |
    # |            180            |         39.467         |
    # |            190            |        42.4034         |
    # |            200            |        68.4522         |
    # |            210            |        52.8405         |
    # |            220            |        50.7969         |
    # |            230            |        43.7583         |
    # |            240            |        42.2165         |
    # |            250            |        40.3607         |
    # |            260            |        44.4572         |
    # |            270            |         50.083         |
    # |            280            |        49.4504         |
    # |            290            |        52.7513         |
    # |            300            |        45.3932         |
    # |            310            |         46.595         |
    # |            320            |        47.7759         |
    # |            330            |        45.7422         |
    # |            340            |        46.9037         |
    # |            350            |        47.1654         |
    # |            360            |         45.917         |
    # |            370            |        46.7888         |
    # |            380            |        46.3741         |
    # |            390            |        46.0772         |
    # |            400            |        46.4491         |
    # +---------------------------+------------------------+

    exp = Experiment(dataset="summer2winter_yosemite", model="cutcamv4",
                     name="BtoA_cutcamv4_sam_7.13.18.24_ttur_warmup_ep400_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets", extra=' --direction BtoA --gpu_ids 0')
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 5 --netF cam_mlp_sample_s'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --stop_idt_epochs 200')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:  83.13658623877538
    # exp.traverse_fid()
    # +---------------------------------+------------------------------+
    # |            epoch_num            |             FID              |
    # +---------------------------------+------------------------------+
    # |                10               |           123.6326           |
    # |                20               |           150.8859           |
    # |                30               |           122.3392           |
    # |                40               |           124.626            |
    # |                50               |           117.9432           |
    # |                60               |           87.9382            |
    # |                70               |           97.9538            |
    # |                80               |           103.1976           |
    # |                90               |           93.1045            |
    # |               100               |           91.2825            |
    # |               110               |           83.7205            |
    # |               120               |            81.492            |
    # |               130               |           84.0645            |
    # |               140               |           91.1626            |
    # |               150               |           95.5121            |
    # |               160               |           81.9702            |
    # |               170               |           87.7282            |
    # |               180               |           117.2741           |
    # |               190               |           82.2292            |
    # |               200               |           84.7156            |
    # |               210               |           88.6684            |
    # |               220               |           83.4287            |
    # |               230               |           80.9151            |
    # |               240               |           75.1543            |
    # |               250               |           84.2089            |
    # |               260               |           77.2133            |
    # |               270               |           74.3935            |
    # |               280               |           76.6665            |
    # |               290               |           71.6117            |
    # |               300               |           76.1227            |
    # |               310               |           76.6816            |
    # |               320               |           78.6024            |
    # |               330               |           78.0282            |
    # |               340               |           80.3753            |
    # |               350               |           79.6773            |
    # |               360               |           82.2985            |
    # |               370               |           82.6983            |
    # |               380               |            82.256            |
    # |               390               |            83.174            |
    # |               400               |           83.1434            |
    # +---------------------------------+------------------------------+

    exp = Experiment(dataset="horse2zebra", model="cutcamv4",
                     name="BtoA_cutcamv4_sam_nosg_7.13.18.24_ttur_warmup_ep400_hid10", load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets", extra=' --direction BtoA')
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 10 --netF cam_mlp_sample_s --stop_gradient False'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --stop_idt_epochs 400')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:  22.07
    # exp.traverse_fid()

    exp = Experiment(dataset="cat2dog", model="cutcamv4",
                     name="BtoA_cutcamv4_sam_nosg_7.13.18.24_ttur_warmup_ep200_hid10", load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets", extra=' --direction BtoA')
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient False'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 10')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:  22.07
    # exp.traverse_fid()
    # +-------------------------------+------------------------+
    # |           epoch_num           |          FID           |
    # +-------------------------------+------------------------+
    # |               10              |        56.0144         |
    # |               20              |        35.3704         |
    # |               30              |         37.524         |
    # |               40              |        37.7624         |
    # |               50              |        31.6279         |
    # |               60              |        37.5056         |
    # |               70              |        37.7654         |
    # |               80              |        37.3736         |
    # |               90              |        35.9032         |
    # |              100              |        33.0574         |
    # |              110              |        33.2638         |
    # |              120              |        35.1555         |
    # |              130              |        30.6351         |
    # |              140              |        28.5612         |
    # |              150              |        29.8871         |
    # |              160              |        26.6832         |
    # |              170              |        24.3947         |
    # |              180              |        23.3061         |
    # |              190              |        22.4687         |
    # |              200              |        22.0772         |
    # +-------------------------------+------------------------+

    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_nosg_7.13.18.24_ttur_warmup_ep200_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient False'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 10')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:  118.1275130682701
    # exp.traverse_fid()
    # FID: 55.963 (time: 0.57 mins)
    # GS: 51.08 (time: 0.47 mins)
    # IS ↗: 0.7905 (time: 0.39 mins)
    # KID: 1.8354 (time: 0.39 mins)
    # PR: 0.7260 / 0.7860 (time: 0.39 mins)
    # MSID: 43.8599 (time: 0.40 mins)
    # +----------------------------+-------------------------+
    # |         epoch_num          |           FID           |
    # +----------------------------+-------------------------+
    # |             10             |         167.5528        |
    # |             20             |         139.3363        |
    # |             30             |         103.6345        |
    # |             40             |         91.4484         |
    # |             50             |         104.0131        |
    # |             60             |         91.7953         |
    # |             70             |         78.6224         |
    # |             80             |          78.836         |
    # |             90             |         79.0837         |
    # |            100             |         70.1946         |
    # |            110             |          77.227         |
    # |            120             |         76.5699         |
    # |            130             |          63.182         |
    # |            140             |         64.9018         |
    # |            150             |         64.5092         |
    # |            160             |         59.0734         |
    # |            170             |         55.6553         |
    # |            180             |         57.9086         |
    # |            190             |          55.516         |
    # |            200             |         55.5775         |
    # +----------------------------+-------------------------+

    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_nosg_7.13.18.24_ttur_ep200_id10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient False'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --stop_idt_epochs 200')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:  118.1275130682701
    # exp.traverse_fid()
    # +------------------------+----------------------+
    # |       epoch_num        |         FID          |
    # +------------------------+----------------------+
    # |           10           |       165.4287       |
    # |           20           |       125.5155       |
    # |           30           |       134.1371       |
    # |           40           |       108.3102       |
    # |           50           |       98.7051        |
    # |           60           |       96.5066        |
    # |           70           |       83.5637        |
    # |           80           |       100.4621       |
    # |           90           |       89.4807        |
    # |          100           |       86.4419        |
    # |          110           |       71.0622        |
    # |          120           |       68.7585        |
    # |          130           |       74.5231        |
    # |          140           |        64.482        |
    # |          150           |       62.2524        |
    # |          160           |       64.4459        |
    # |          170           |       60.2525        |
    # |          180           |       61.5764        |
    # |          190           |       61.7847        |
    # |          200           |       60.9903        |
    # +------------------------+----------------------+

    exp = Experiment(dataset="IXI", model="cutcamv4", name="cutcamv4_sam_3,7,12,14,17,19,24,28_ttur_ep400_noid",
                     load_size=256, gpu_ids=0,
                     input_nc=1, output_nc=1, dataroot=r"../datasets")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,12,14,17,19,24,28 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient False '
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --stop_idt_epochs 400')
    # exp.test(extra=' --batch_size 1')
    # exp.eval(metric_list, testB=False)  # 20.8269 | 0.6992

    # exp.fid()  # FID: 44.54072728978899

    exp = Experiment(dataset="horse2zebra", model="cutcamv4", name="cutcamv4_sam_4,8,12,17,21,25_nosg_ttur_ep400_id10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 4,8,12,17,21,25 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient False '
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --stop_idt_epochs 400')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID: 44.54072728978899

    exp = Experiment(dataset="vangogh2photo", model="cutcamv4",
                     name="cutcamv4_sam_3.7.13.18.24.28_nosg_ttur_ep200_hid10", load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient False'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(epoch='latest', extra=' --batch_size 8')
    # exp.fid(epoch='latest')  # FID:
    # metric_list = ['dists', 'lpips']
    # exp.eval(metric_list)
    # exp.traverse_fid()

    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_3.7.13.18.24.28_nosg_ttur_ep200_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient False'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # GS ↗: 44.7464 (time: 0.56 mins)
    # KID: 2.6303 (time: 0.38 mins)
    # PR: 0.6820 / 0.7400 (time: 0.40 mins)
    # MSID: 84.2733 (time: 0.41 mins)
    # +---------------------------+------------------------+
    # |         epoch_num         |          FID           |
    # +---------------------------+------------------------+
    # |             10            |        168.0016        |
    # |             20            |        146.3059        |
    # |             30            |        134.5255        |
    # |             40            |        128.5984        |
    # |             50            |        117.4238        |
    # |             60            |        97.6826         |
    # |             70            |        96.3849         |
    # |             80            |        87.2256         |
    # |             90            |        83.7964         |
    # |            100            |        85.6802         |
    # |            110            |        78.8407         |
    # |            120            |        83.2055         |
    # |            130            |        66.5716         |
    # |            140            |        69.9994         |
    # |            150            |        65.4993         |
    # |            160            |        65.7262         |
    # |            170            |        65.1042         |
    # |            180            |        61.7972         |
    # |            190            |        62.8835         |
    # |            200            |        62.4746         |
    # +---------------------------+------------------------+

    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_3.7.13.18.24.28_ttur_ep200_hid10",
                     load_size=256, gpu_ids='0',
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=True,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --epoch_count 101'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +-------------------------+----------------------+
    # |        epoch_num        |         FID          |
    # +-------------------------+----------------------+
    # |            10           |       167.9927       |
    # |            20           |       139.3973       |
    # |            30           |       130.7672       |
    # |            40           |       109.6543       |
    # |            50           |       109.2466       |
    # |            60           |       114.6073       |
    # |            70           |       94.9398        |
    # |            80           |        91.377        |
    # |            90           |       110.2256       |
    # |           100           |       87.7835        |
    # |           110           |       83.0366        |
    # |           120           |       78.5991        |
    # |           130           |       86.9684        |
    # |           140           |       81.7619        |
    # |           150           |        68.505        |
    # |           160           |       66.3358        |
    # |           170           |       66.2387        |
    # |           180           |       63.7238        |
    # |           190           |       64.4375        |
    # |           200           |        64.618        |
    # +-------------------------+----------------------+

    exp = Experiment(dataset="cityscapes", model="cutcamv4", name="cutcamv4_sam_3.7.13.18.24.28_ttur_ep400_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets", extra=' --direction BtoA')
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # exp.test(extra=' --batch_size 8')
    # exp.eval_cityscapes()

    exp = Experiment(dataset="cityscapes", model="cutcamv4", name="cutcamv4_sam_3.7.13.18.24.28_ttur_ep400_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets", extra=' --direction BtoA')
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # exp.test(extra=' --batch_size 8')
    # exp.eval_cityscapes()
    # [2022-07-16 18:57:33,777 segment.py:581 test_more] ===> mAP 23.360
    # [2022-07-16 18:57:33,777 segment.py:583 test_more] ===> pixAcc 66.785
    # [2022-07-16 18:57:33,777 segment.py:584 test_more] ===> classAcc 30.846
    # [2022-07-16 18:57:33,778 segment.py:585 test_more] ===> mIoU 23.364
    # +--------------------------+-----------------------+
    # |            10            |        248.4128       |
    # |            20            |        200.8033       |
    # |            30            |        185.0225       |
    # |            40            |        169.4563       |
    # |            50            |        116.2048       |
    # |            60            |        92.3037        |
    # |            70            |        82.6408        |
    # |            80            |        90.7989        |
    # |            90            |        67.1864        |
    # |           100            |        72.4006        |
    # |           110            |        82.4197        |
    # |           120            |        63.7349        |
    # |           130            |        62.5815        |
    # |           140            |        59.7328        |
    # |           150            |        54.9479        |
    # |           160            |        58.9616        |
    # |           170            |        71.2152        |
    # |           180            |        71.2156        |
    # |           190            |        71.1676        |
    # |           200            |        53.9914        |
    # |           210            |        59.4215        |
    # |           220            |        55.5127        |
    # |           230            |        52.2052        |
    # |           240            |        51.2559        |
    # |           250            |        54.6883        |
    # |           260            |        50.9591        |
    # |           270            |        49.6073        |
    # |           280            |        53.7264        |
    # |           290            |        51.1537        |
    # |           300            |        51.8294        |
    # |           310            |        50.2498        |
    # |           320            |        49.6962        |
    # |           330            |        50.0953        |
    # |           340            |        49.5805        |
    # |           350            |        49.0371        |
    # |           360            |        48.0714        |
    # |           370            |        47.6456        |
    # |           380            |        48.0735        |
    # |           390            |        47.6424        |
    # |           400            |        47.8531        |
    # +--------------------------+-----------------------+

    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_3.7.13.18.24.28_ttur_ep200_noid",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=False, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +------------------------+----------------------+
    # |       epoch_num        |         FID          |
    # +------------------------+----------------------+
    # |           10           |       274.9842       |
    # |           20           |       233.4331       |
    # |           30           |       207.9237       |
    # |           40           |       222.3842       |
    # |           50           |       296.9033       |
    # |           60           |       218.0342       |
    # |           70           |       250.896        |
    # |           80           |       226.3043       |
    # |           90           |       249.8047       |
    # |          100           |       213.618        |
    # |          110           |       179.5714       |
    # |          120           |       248.821        |
    # |          130           |       207.3893       |
    # |          140           |       236.0742       |
    # |          150           |       217.8948       |
    # |          160           |       234.9718       |
    # |          170           |       238.1982       |
    # |          180           |       249.0376       |
    # |          190           |       235.8255       |
    # |          200           |       181.7818       |
    # +------------------------+----------------------+

    # Training completed in 20.0hrs, 7.0min, 27.1sec!
    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_os4_is0.75_7.13.18.24_ttur_ep200_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --oversample_ratio 4 --random_ratio 0.25'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +---------------------------+------------------------+
    # |         epoch_num         |          FID           |
    # +---------------------------+------------------------+
    # |             10            |        172.8504        |
    # |             20            |        128.0283        |
    # |             30            |        118.3153        |
    # |             40            |        114.4524        |
    # |             50            |        116.0608        |
    # |             60            |        109.9949        |
    # |             70            |        111.7768        |
    # |             80            |        88.1669         |
    # |             90            |        90.9697         |
    # |            100            |        89.3408         |
    # |            110            |        95.8964         |
    # |            120            |         99.677         |
    # |            130            |        79.5386         |
    # |            140            |        70.8252         |
    # |            150            |        70.8823         |
    # |            160            |        69.8993         |
    # |            170            |        66.4989         |
    # |            180            |        66.5294         |
    # |            190            |        64.6609         |
    # |            200            |        65.9318         |
    # +---------------------------+------------------------+

    # Training completed in 20.0hrs, 7.0min, 27.1sec!
    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_os2_7.13.18.24_ttur_ep200_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --oversample_ratio 2'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +------------------------+----------------------+
    # |       epoch_num        |         FID          |
    # +------------------------+----------------------+
    # |           10           |       183.4903       |
    # |           20           |       150.7593       |
    # |           30           |       115.3015       |
    # |           40           |       117.2966       |
    # |           50           |       114.778        |
    # |           60           |       99.9484        |
    # |           70           |       89.5161        |
    # |           80           |       101.4846       |
    # |           90           |       85.2165        |
    # |          100           |       87.8683        |
    # |          110           |       90.6747        |
    # |          120           |       88.9273        |
    # |          130           |       81.0635        |
    # |          140           |       67.0813        |
    # |          150           |       64.8196        |
    # |          160           |       67.9118        |
    # |          170           |       61.6391        |
    # |          180           |       63.3763        |
    # |          190           |       62.8419        |
    # |          200           |       62.3936        |
    # +------------------------+----------------------+

    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_os4_7.12.19.24_ttur_ep200_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,12,19,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --oversample_ratio 4 '
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +------------------------+----------------------+
    # |       epoch_num        |         FID          |
    # +------------------------+----------------------+
    # |           10           |       190.7024       |
    # |           20           |       146.6424       |
    # |           30           |       128.1846       |
    # |           40           |       114.0186       |
    # |           50           |       124.189        |
    # |           60           |       106.3073       |
    # |           70           |       96.5025        |
    # |           80           |       93.3987        |
    # |           90           |       83.3027        |
    # |          100           |        82.769        |
    # |          110           |       97.0332        |
    # |          120           |       80.1669        |
    # |          130           |       82.1186        |
    # |          140           |       73.1057        |
    # |          150           |       73.3072        |
    # |          160           |       70.9247        |
    # |          170           |       71.0461        |
    # |          180           |       66.2599        |
    # |          190           |        63.437        |
    # |          200           |       62.9041        |
    # +------------------------+----------------------+

    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_os2_7.13.18.24_ttur_ep200_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --oversample_ratio 2 '
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()

    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_nlsam_7.13.18.24_ttur_ep200_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_nls'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +-----------------------+---------------------+
    # |       epoch_num       |         FID         |
    # +-----------------------+---------------------+
    # |           10          |       168.5852      |
    # |           20          |       149.0254      |
    # |           30          |       128.5368      |
    # |           40          |       115.8919      |
    # |           50          |       108.4633      |
    # |           60          |       121.6374      |
    # |           70          |       98.8937       |
    # |           80          |       91.1393       |
    # |           90          |       91.2528       |
    # |          100          |       113.5104      |
    # |          110          |       104.4608      |
    # |          120          |       117.5291      |
    # |          130          |       90.6136       |
    # |          140          |       84.4995       |
    # |          150          |       88.0419       |
    # |          160          |       79.7423       |
    # |          170          |       76.0284       |
    # |          180          |       69.9816       |
    # |          190          |       71.9751       |
    # |          200          |       72.3031       |
    # +-----------------------+---------------------+

    # Training completed in 23.0hrs, 9.0min, 14.4sec!
    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_7.13.18.24_ttur_ep200_id10", load_size=256,
                     gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_idt_epochs 200'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()
    # exp.traverse_fid()
    # +----------------------+--------------------+
    # |      epoch_num       |        FID         |
    # +----------------------+--------------------+
    # |          10          |      175.5952      |
    # |          20          |      157.4797      |
    # |          30          |      124.1904      |
    # |          40          |      131.5795      |
    # |          50          |      108.1256      |
    # |          60          |      111.0603      |
    # |          70          |      91.9648       |
    # |          80          |      94.5987       |
    # |          90          |      83.6223       |
    # |         100          |      95.2736       |
    # |         110          |      78.1661       |
    # |         120          |      90.4731       |
    # |         130          |      79.2993       |
    # |         140          |      76.1148       |
    # |         150          |      72.0064       |
    # |         160          |      70.2072       |
    # |         170          |      68.6106       |
    # |         180          |      71.1268       |
    # |         190          |      67.0803       |
    # |         200          |      69.5551       |
    # +----------------------+--------------------+

    # Training completed in 21.0hrs, 41.0min, 24.7sec!
    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_nosg_7.13.18.24_ttur_ep200_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient False'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()
    # exp.traverse_fid()
    # +-------------------------+----------------------+
    # |        epoch_num        |         FID          |
    # +-------------------------+----------------------+
    # |            10           |       168.7081       |
    # |            20           |       137.3605       |
    # |            30           |       102.2299       |
    # |            40           |       118.0697       |
    # |            50           |       96.6602        |
    # |            60           |       111.4882       |
    # |            70           |       106.3192       |
    # |            80           |       85.8176        |
    # |            90           |       82.5233        |
    # |           100           |       96.5551        |
    # |           110           |       99.0485        |
    # |           120           |       89.6325        |
    # |           130           |       72.0179        |
    # |           140           |       65.7917        |
    # |           150           |       66.1705        |
    # |           160           |       63.8823        |
    # |           170           |       62.3738        |
    # |           180           |       57.9158        |
    # |           190           |       59.3073        |
    # |           200           |       58.8406        |
    # +-------------------------+----------------------+

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutcamv5", name="cutcamv5_sam_7.13.18.24_ttur_ep200_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # GS ↗: 48.1062 (time: 0.50 mins)
    # KID: 2.9442 (time: 0.35 mins)
    # PR: 0.6720 / 0.6900 (time: 0.36 mins)
    # MSID: 45.7432 (time: 0.35 mins)
    # +-----------------------+---------------------+
    # |       epoch_num       |         FID         |
    # +-----------------------+---------------------+
    # |           10          |       176.8606      |
    # |           20          |       154.4989      |
    # |           30          |       127.2415      |
    # |           40          |       119.5553      |
    # |           50          |       101.2122      |
    # |           60          |       102.0113      |
    # |           70          |       116.4491      |
    # |           80          |       92.2394       |
    # |           90          |       88.9421       |
    # |          100          |        90.399       |
    # |          110          |       98.5278       |
    # |          120          |       84.7091       |
    # |          130          |       80.4357       |
    # |          140          |       73.1637       |
    # |          150          |       72.0422       |
    # |          160          |       74.0184       |
    # |          170          |       66.4126       |
    # |          180          |        65.881       |
    # |          190          |       66.4726       |
    # |          200          |       65.9544       |
    # +-----------------------+---------------------+

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutv4", name="cutv4_sam_7.13.18.24_ttur_ep200_hid10", load_size=256,
                     gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF mlp_sample'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # GS ↗: 58.4936 (time: 0.48 mins)
    # KID: 2.6322 (time: 0.35 mins)
    # PR: 0.7340 / 0.7140 (time: 0.34 mins)
    # MSID: 55.4269 (time: 0.33 mins)
    # +----------------------+--------------------+
    # |      epoch_num       |        FID         |
    # +----------------------+--------------------+
    # |          10          |      158.4682      |
    # |          20          |      137.6165      |
    # |          30          |      114.2361      |
    # |          40          |      109.1035      |
    # |          50          |      95.3414       |
    # |          60          |      99.4407       |
    # |          70          |      98.1007       |
    # |          80          |      104.0953      |
    # |          90          |       78.666       |
    # |         100          |      85.6234       |
    # |         110          |      97.7987       |
    # |         120          |      83.4818       |
    # |         130          |      87.3283       |
    # |         140          |       74.418       |
    # |         150          |      77.1789       |
    # |         160          |      65.3653       |
    # |         170          |      67.1704       |
    # |         180          |      62.3757       |
    # |         190          |      63.4388       |
    # |         200          |      63.7834       |
    # +----------------------+--------------------+

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_7.13.18.24_ttur_ep200_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +-----------------------+---------------------+
    # |       epoch_num       |         FID         |
    # +-----------------------+---------------------+
    # |           10          |       161.6222      |
    # |           20          |       125.8977      |
    # |           30          |       113.3089      |
    # |           40          |       127.6363      |
    # |           50          |       95.5686       |
    # |           60          |       104.8839      |
    # |           70          |       98.8334       |
    # |           80          |       86.5752       |
    # |           90          |       83.8108       |
    # |          100          |       88.8134       |
    # |          110          |       76.9745       |
    # |          120          |       82.8539       |
    # |          130          |       77.2542       |
    # |          140          |       68.1658       |
    # |          150          |       67.5319       |
    # |          160          |       64.9628       |
    # |          170          |       60.4214       |
    # |          180          |       59.4916       |
    # |          190          |       58.9412       |
    # |          200          |       58.0041       |
    # +-----------------------+---------------------+

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_7,24_ttur_ep200_hid10", load_size=256,
                     gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=1, n_epochs_decay=100, nce_idt=True, continue_train=True,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +--------------------+------------------+
    # |     epoch_num      |       FID        |
    # +--------------------+------------------+
    # |         10         |     133.3965     |
    # |         20         |     84.2234      |
    # |         30         |     104.7609     |
    # |         40         |     101.7381     |
    # |         50         |     88.7442      |
    # |         60         |      75.248      |
    # |         70         |     72.1483      |
    # |         80         |     75.1979      |
    # |         90         |     66.9389      |
    # |        100         |     67.6216      |
    # +--------------------+------------------+

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_7,11,16,24_ttur_ep200_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,11,16,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +-----------------------+---------------------+
    # |       epoch_num       |         FID         |
    # +-----------------------+---------------------+
    # |           10          |       161.8368      |
    # |           20          |       153.0048      |
    # |           30          |       137.4609      |
    # |           40          |       114.2848      |
    # |           50          |       96.7786       |
    # |           60          |       116.5598      |
    # |           70          |       91.0205       |
    # |           80          |       94.1683       |
    # |           90          |       88.3646       |
    # |          100          |       97.0036       |
    # |          110          |       101.0179      |
    # |          120          |       92.2928       |
    # |          130          |        79.105       |
    # |          140          |       73.7553       |
    # |          150          |       75.7205       |
    # |          160          |        67.846       |
    # |          170          |       63.8534       |
    # |          180          |       68.9561       |
    # |          190          |       64.1943       |
    # |          200          |       64.6033       |
    # +-----------------------+---------------------+

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_7.13.18.24_ttur_ep200_id0", load_size=256,
                     gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=False, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()

    # supernet training
    exp = Experiment(dataset="cityscapes", model="cutcamv4", name="cutcamv4_sam_7.13.18.24_ttur_ep400_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets", extra=' --direction BtoA')
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +------------------------+----------------------+
    # |       epoch_num        |         FID          |
    # +------------------------+----------------------+
    # |           10           |       261.0059       |
    # |           20           |       214.6536       |
    # |           30           |        208.77        |
    # |           40           |       205.0731       |
    # |           50           |       129.0013       |
    # |           60           |       125.996        |
    # |           70           |       112.6222       |
    # |           80           |       86.6787        |
    # |           90           |       90.2954        |
    # |          100           |       72.4519        |
    # +------------------------+----------------------+

    # supernet training
    exp = Experiment(dataset="horse2zebra", model="cutcamv4", name="cutcamv4_sam_7.13.18.24_ttur_ep400_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +------------------------+----------------------+
    # |       epoch_num        |         FID          |
    # +------------------------+----------------------+
    # |           10           |       149.2642       |
    # |           20           |       137.7932       |
    # |           30           |       101.5431       |
    # |           40           |       119.0623       |
    # |           50           |       83.8481        |
    # |           60           |       70.0164        |
    # |           70           |       99.1478        |
    # |           80           |       56.0033        |
    # |           90           |       40.1052        |
    # |          100           |       46.7408        |
    # |          110           |       46.2339        |
    # |          120           |       76.2245        |
    # |          130           |       47.7762        |
    # |          140           |       59.9873        |
    # |          150           |       44.6828        |
    # |          160           |       68.0022        |
    # |          170           |       51.6609        |
    # |          180           |       37.5327        |
    # |          190           |       42.5388        |
    # |          200           |       57.1207        |
    # |          210           |       46.4263        |
    # |          220           |       83.7578        |
    # |          230           |       41.2419        |
    # |          240           |       41.5643        |
    # |          250           |       38.2186        |
    # |          260           |       35.8878        |
    # |          270           |        35.169        |
    # |          280           |       39.9275        |
    # |          290           |       48.7834        |
    # |          300           |       44.0522        |
    # |          310           |       39.7777        |
    # |          320           |       42.4141        |
    # |          330           |       39.3515        |
    # |          340           |       43.1399        |
    # |          350           |       44.0175        |
    # |          360           |       40.5586        |
    # |          370           |        41.82         |
    # |          380           |        40.515        |
    # |          390           |       42.4676        |
    # |          400           |        42.559        |
    # +------------------------+----------------------+

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_7,13,18,24_ttur_ep200_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=0, n_epochs_decay=100, nce_idt=False, continue_train=True,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +------------------------+-------------------+
    # |       epoch_num        |        FID        |
    # +------------------------+-------------------+
    # |           10           |      91.0928      |
    # |           20           |      95.5623      |
    # |           30           |      91.1417      |
    # |           40           |      69.4653      |
    # |           50           |      74.0996      |
    # |           60           |      66.9104      |
    # |           70           |      61.1206      |
    # |           80           |      60.4439      |
    # |           90           |      61.5136      |
    # |          100           |      60.8186      |
    # +------------------------+-------------------+

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_12,19_ep200_widt", load_size=256,
                     gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 12,19 --display_freq 100'
    #                 ' --lambda_IDT 1 --lambda_NCE 5 --netF cam_mlp_sample_s'
    #                 ' --lr_G 2e-4 --lr_F 2e-4 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +-------------------+-----------------+
    # |     epoch_num     |       FID       |
    # +-------------------+-----------------+
    # |         50        |     108.5392    |
    # |        100        |     86.0869     |
    # |        150        |      66.694     |
    # |        200        |      64.557     |
    # +-------------------+-----------------+

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_ep200_widt", load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(epoch='150', extra=' --batch_size 8')
    # exp.fid(epoch='150')  # FID:  69.19440187440122  , Training completed in 22.0hrs, 43.0min, 17.1sec!
    # exp.traverse_fid()

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutv5", name="cutv5_ep200", load_size=286, gpu_ids=0,
                     netG='resnet_9blocks', input_nc=3, output_nc=3, dataroot=r"../datasets",
                     extra=f" --ngf 64 --n_blocks 9 --nce_layers 12,20"
                           f" --lambda_IDT 1 --lambda_NCE 2")
    # exp.train(batch_size=4, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --display_freq 100'
    #                 ' --gan_mode ralsgan')
    # exp.test()
    # exp.fid()  # ep135: 72.47, ep200: 64.3627, Training completed in 23.0hrs, 3.0min, 10.9sec!

    # supernet training
    # exp = Experiment(dataset="cat2dog", model="cutcamv5", name="cutcamv5_sa_ep200_widt", load_size=286, gpu_ids=0,
    #                  netG='sa', input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False, netD='sa',
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,24 --display_freq 100'
    #                 ' --lambda_IDT 1 --lambda_NCE 2 --netF cam_mlp_sample --gan_mode ralsgan')
    # exp.test()
    # exp.fid()  # ep121: 77.7655, ep200: 62.4467, Training completed in 18.0hrs, 15.0min, 7.7sec!

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutcamv3", name="cutcamv3_ep200_widt", load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_nl'
    #                 ' --lr_G 2e-4 --lr_F 2e-4 --lr_D 2e-4')
    # exp.test()
    # exp.fid()  # FID:  65.0003706038824, Training completed in 22.0hrs, 58.0min, 23.4sec!

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_ep200_widt", load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,24 --display_freq 100'
    #                 ' --lambda_IDT 1 --lambda_NCE 2 --netF cam_mlp_sample_nl'
    #                 ' --lr_G 2e-4 --lr_F 2e-4 --lr_D 2e-4')
    # exp.test()
    # exp.fid()  # ep121: 77.7655, ep200: 62.4467, Training completed in 18.0hrs, 15.0min, 7.7sec!

    # supernet training
    # exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_swin_ttur_ep200_widt", load_size=286, gpu_ids=0,
    #                   input_nc=3, output_nc=3, dataroot=r"../datasets", netG='swin')
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,22 --display_freq 100'
    #                 ' --lambda_IDT 1 --lambda_NCE 2 --netF cam_mlp_sample'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test()
    # exp.fid()  # ep121: 77.7655, ep200: 62.4467, Training completed in 18.0hrs, 15.0min, 7.7sec!

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutv3", name="cutv3_ep200", load_size=286,
                     netG='resnet_9blocks', input_nc=3, output_nc=3, dataroot=r"../datasets",
                     extra=f" --ngf 64 --n_blocks 9 --nce_layers 7,12,20,24"
                           f" --lambda_IDT 1 --lambda_NCE 2 --gpu_ids 1")
    # exp.train(batch_size=4, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --display_freq 100')
    # exp.test()
    # exp.fid()  # ep135: 74, ep200: 70.06

    show_results(metric_list, rowsA, rowsB)


def s2c_city():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'vsi', 'total_variation']
    # metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    exp = Experiment(dataset="cityscapes", model="enco", name="enco_3,10,13,18,21,28_ttur_warmup_ep400_id10",
                     load_size=256, gpu_ids='0', input_nc=3, output_nc=3,
                     dataroot="/home/yht/Casit/Datasets/ez/datasets", )
    exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False, netD='sa',
              extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,10,13,18,21,28 --display_freq 100'
                    ' --stop_idt_epochs 400'
                    ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_gradient True --gan_mode lsgan'
                    ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 20 --flip_equivariance True'
                    ' --num_patches 256')
    exp.test(extra=' --batch_size 8', epoch='latest')
    exp.fid(epoch='latest')  # FID:
    exp.traverse_fid()
    exp.eval_cityscapes()

    # supernet training
    exp = Experiment(dataset="cityscapes", model="cutcamv4", name="cutcamv4_ssa_7,13,18,24_ttur_ep400_wid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets", extra=' --direction BtoA', netG='sa')
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False, netD='sa',
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s --stop_idt_epochs 400'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID: 52
    # exp.traverse_fid()
    # exp.test(extra=' --batch_size 8')
    # exp.eval_cityscapes()
    # +------------------------+----------------------+
    # |       epoch_num        |         FID          |
    # +------------------------+----------------------+
    # |           10           |       216.6291       |
    # |           20           |       123.9782       |
    # |           30           |       178.1847       |
    # |           40           |       87.5072        |
    # |           50           |       84.8139        |
    # |           60           |       76.8791        |
    # |           70           |       83.2311        |
    # |           80           |        93.532        |
    # |           90           |       70.0539        |
    # |          100           |       84.8678        |
    # |          110           |        62.237        |
    # |          120           |       58.2742        |
    # |          130           |        66.318        |
    # |          140           |       58.8671        |
    # |          150           |       58.0549        |
    # |          160           |       58.6911        |
    # |          170           |       60.8072        |
    # |          180           |       64.5296        |
    # |          190           |       54.4096        |
    # |          200           |       57.2732        |
    # |          210           |       60.2185        |
    # |          220           |       55.5325        |
    # |          230           |       52.8568        |
    # |          240           |       55.1839        |
    # |          250           |       57.0593        |
    # |          260           |       52.6168        |
    # |          270           |       52.9651        |
    # |          280           |       53.5546        |
    # |          290           |       52.4902        |
    # |          300           |        48.643        |
    # |          310           |       52.6733        |
    # |          320           |       49.6655        |
    # |          330           |       50.5122        |
    # |          340           |       49.3005        |
    # |          350           |       47.5818        |
    # |          360           |       48.5087        |
    # |          370           |       48.6211        |
    # |          380           |       48.3385        |
    # |          390           |       48.0417        |
    # |          400           |       48.0195        |
    # +------------------------+----------------------+

    # supernet training
    exp = Experiment(dataset="cityscapes", model="cutcamv4", name="cutcamv4_sam_7.13.18.24_ttur_ep400_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets", extra=' --direction BtoA')
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID: 52
    # exp.traverse_fid()
    # exp.eval_cityscapes()
    # +------------------------+----------------------+
    # |       epoch_num        |         FID          |
    # +------------------------+----------------------+
    # |           10           |       261.0059       |
    # |           20           |       214.6536       |
    # |           30           |        208.77        |
    # |           40           |       205.0731       |
    # |           50           |       129.0013       |
    # |           60           |       125.996        |
    # |           70           |       112.6222       |
    # |           80           |       86.6787        |
    # |           90           |       90.2954        |
    # |          100           |       72.4519        |
    # +------------------------+----------------------+

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_7,13,18,24_ttur_ep200_hid10",
                     load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=0, n_epochs_decay=100, nce_idt=False, continue_train=True,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,13,18,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +------------------------+-------------------+
    # |       epoch_num        |        FID        |
    # +------------------------+-------------------+
    # |           10           |      91.0928      |
    # |           20           |      95.5623      |
    # |           30           |      91.1417      |
    # |           40           |      69.4653      |
    # |           50           |      74.0996      |
    # |           60           |      66.9104      |
    # |           70           |      61.1206      |
    # |           80           |      60.4439      |
    # |           90           |      61.5136      |
    # |          100           |      60.8186      |
    # +------------------------+-------------------+

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_12,19_ep200_widt", load_size=256,
                     gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 12,19 --display_freq 100'
    #                 ' --lambda_IDT 1 --lambda_NCE 5 --netF cam_mlp_sample_s'
    #                 ' --lr_G 2e-4 --lr_F 2e-4 --lr_D 2e-4')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:
    # exp.traverse_fid()
    # +-------------------+-----------------+
    # |     epoch_num     |       FID       |
    # +-------------------+-----------------+
    # |         50        |     108.5392    |
    # |        100        |     86.0869     |
    # |        150        |      66.694     |
    # |        200        |      64.557     |
    # +-------------------+-----------------+

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_sam_ep200_widt", load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_s'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test(epoch='150', extra=' --batch_size 8')
    # exp.fid(epoch='150')  # FID:  69.19440187440122  , Training completed in 22.0hrs, 43.0min, 17.1sec!
    # exp.traverse_fid()

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutv5", name="cutv5_ep200", load_size=286, gpu_ids=0,
                     netG='resnet_9blocks', input_nc=3, output_nc=3, dataroot=r"../datasets",
                     extra=f" --ngf 64 --n_blocks 9 --nce_layers 12,20"
                           f" --lambda_IDT 1 --lambda_NCE 2")
    # exp.train(batch_size=4, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --display_freq 100'
    #                 ' --gan_mode ralsgan')
    # exp.test()
    # exp.fid()  # ep135: 72.47, ep200: 64.3627, Training completed in 23.0hrs, 3.0min, 10.9sec!

    # supernet training
    # exp = Experiment(dataset="cat2dog", model="cutcamv5", name="cutcamv5_sa_ep200_widt", load_size=286, gpu_ids=0,
    #                  netG='sa', input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False, netD='sa',
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,24 --display_freq 100'
    #                 ' --lambda_IDT 1 --lambda_NCE 2 --netF cam_mlp_sample --gan_mode ralsgan')
    # exp.test()
    # exp.fid()  # ep121: 77.7655, ep200: 62.4467, Training completed in 18.0hrs, 15.0min, 7.7sec!

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutcamv3", name="cutcamv3_ep200_widt", load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,24 --display_freq 100'
    #                 ' --lambda_IDT 10 --lambda_NCE 2 --netF cam_mlp_sample_nl'
    #                 ' --lr_G 2e-4 --lr_F 2e-4 --lr_D 2e-4')
    # exp.test()
    # exp.fid()  # FID:  65.0003706038824, Training completed in 22.0hrs, 58.0min, 23.4sec!

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_ep200_widt", load_size=256, gpu_ids=0,
                     input_nc=3, output_nc=3, dataroot=r"../datasets")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,24 --display_freq 100'
    #                 ' --lambda_IDT 1 --lambda_NCE 2 --netF cam_mlp_sample_nl'
    #                 ' --lr_G 2e-4 --lr_F 2e-4 --lr_D 2e-4')
    # exp.test()
    # exp.fid()  # ep121: 77.7655, ep200: 62.4467, Training completed in 18.0hrs, 15.0min, 7.7sec!

    # supernet training
    # exp = Experiment(dataset="cat2dog", model="cutcamv4", name="cutcamv4_swin_ttur_ep200_widt", load_size=286, gpu_ids=0,
    #                   input_nc=3, output_nc=3, dataroot=r"../datasets", netG='swin')
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 7,22 --display_freq 100'
    #                 ' --lambda_IDT 1 --lambda_NCE 2 --netF cam_mlp_sample'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4')
    # exp.test()
    # exp.fid()  # ep121: 77.7655, ep200: 62.4467, Training completed in 18.0hrs, 15.0min, 7.7sec!

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutv3", name="cutv3_ep200", load_size=286,
                     netG='resnet_9blocks', input_nc=3, output_nc=3, dataroot=r"../datasets",
                     extra=f" --ngf 64 --n_blocks 9 --nce_layers 7,12,20,24"
                           f" --lambda_IDT 1 --lambda_NCE 2 --gpu_ids 1")
    # exp.train(batch_size=4, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --display_freq 100')
    # exp.test()
    # exp.fid()  # ep135: 74, ep200: 70.06

    # # supernet training
    # exp = Experiment(dataset="cat2dog", model="dcl", name="dcl_swin_ep200", load_size=286,
    #                  netG='swin', input_nc=3, output_nc=3, dataroot=r"../datasets",
    #                  extra=f" --ngf 64 --n_blocks 9 --nce_layers 1,4,8,12,16")
    # exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
    # extra=' --save_latest_freq 5000')
    # exp.test()
    # exp.fid()

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cut", name="cut_swin_unet_ep200", load_size=286,
                     netG='swin', input_nc=3, output_nc=3, dataroot=r"../datasets",
                     extra=f" --ngf 64 --n_blocks 9")
    # exp.train(batch_size=4, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False, netD='unet',
    #           extra=' --save_latest_freq 5000 --verbose')
    # exp.test()
    # exp.fid()

    # 1546 sec, ~0.5h
    exp = CycleGAN(dataset="IXI", model="cycle_gan", name="cyclegan_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cut", name="cut_ep10", load_size=256, netG='swin',
                     extra=' --n_blocks 9')
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="dcl", name="dcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="simdcl", name="simdcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False,)
    #           # extra=" --nce_includes_all_negatives_from_minibatch True")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    show_results(metric_list, rowsA, rowsB)


def baseline_pretrained():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'vsi', 'total_variation']
    # metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    ## cat2dog
    exp = Experiment(dataset="horse2zebra_tmp", model="dcl", name="dcl_BtoA_pretrained", load_size=256,
                     input_nc=3, output_nc=3, dataroot="../datasets")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    ## cat2dog
    exp = Experiment(dataset="cat2dog", model="cut", name="cut_pretrained", load_size=256,
                     input_nc=3, output_nc=3, dataroot="../datasets", )
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    ## cat2dog
    exp = Experiment(dataset="cityscapes", model="cut", name="cut_pretrained", load_size=256,
                     input_nc=3, output_nc=3, dataroot="../datasets", )
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=False, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # show_results(metric_list, rowsA, rowsB)


def baseline_dog2cat():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'vsi', 'total_variation']
    # metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    # 1546 sec, ~0.5h
    exp = CycleGAN(dataset="dog2cat", model="cycle_gan", name="cyclegan_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, lambda_identity=0, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="dog2cat", model="cut", name="cut_ep200", load_size=286, input_nc=3, output_nc=3,
                     dataroot="../datasets")
    exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False)
    exp.test()
    exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="dcl", name="dcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False)
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="simdcl", name="simdcl_ep10", load_size=256)
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, input_nc=1, nce_idt=True, continue_train=False,)
    #           # extra=" --nce_includes_all_negatives_from_minibatch True")
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    show_results(metric_list, rowsA, rowsB)


def exp_cut_spos_cityscapes():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'vsi', 'total_variation']
    # metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    # 1546 sec, ~0.5h
    exp = Experiment(dataset="cityscapes", model="cutspos", name="cutspos_ep300", load_size=256, netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=" --direction BtoA --ngf 64 --n_blocks 9 --n_choices 3")
    # exp.train(batch_size=2, n_epochs=150, n_epochs_decay=150, nce_idt=False, continue_train=False, netD='basic',
    #           extra=' --lambda_NCE 1 --save_latest_freq 5000 --display_freq 100')
    choice = str((1, 0, 2, 1, 2, 2, 0, 2, 1)).replace(' ', '')
    # exp.test(extra=f' --choice_spos {choice} --batch_size 6')

    exp = Experiment(dataset="cityscapes", model="cutspos", name=f"cutspos_ep300_{choice[1:-1].replace(',', '')}_ep100",
                     load_size=256, netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=f" --direction BtoA --choice_spos {choice} --ngf 64 --n_blocks 9 --n_choices 3")
    exp.train(batch_size=2, n_epochs=0, n_epochs_decay=100, nce_idt=False, continue_train=True, netD='basic',
              extra=' --lambda_NCE 1 --save_latest_freq 5000')
    # ' --pretrained_name cityscapes_cutspos_ep300')
    exp.test(extra=f' --choice_spos {choice}')
    exp.fid()
    # exp.eval(metric_list, testB=False)

    # show_results(metric_list, rowsA, rowsB)


def exp_cut_spos_horse2zebra():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'vsi', 'total_variation']
    # metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    # 1546 sec, ~0.5h
    exp = Experiment(dataset="horse2zebra", model="simcutspos", name="cutspos_ep200", load_size=286, netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=" --ngf 64 --n_blocks 9 --n_choices 2 --gpu_ids 1")
    exp.train(batch_size=3, n_epochs=200, n_epochs_decay=0, nce_idt=False, continue_train=False, netD='basic',
              extra=' --lambda_NCE 1 --save_latest_freq 5000 --display_freq 100')

    # 1546 sec, ~0.5h
    exp = Experiment(dataset="horse2zebra", model="cutspos", name="cutspos_ep400", load_size=286, netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 5,9,12,16")
    # exp.train(batch_size=2, n_epochs=200, n_epochs_decay=200, nce_idt=False, continue_train=False, netD='basic_spos',
    #           extra=' --lambda_NCE 2 --save_latest_freq 5000')
    choice_G = str((3, 2, 2, 1, 3, 0, 2, 3, 1)).replace(' ', '')
    choice_D = str((1, 0, 1, 0, 2)).replace(' ', '')
    exp = Experiment(dataset="horse2zebra", model="cutspos", name="cutspos_ep400_212332011_ep100", load_size=256,
                     netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=f" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 5,9,12,16"
                           f" --choice_G {choice_G} --choice_D {choice_D}")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=100, nce_idt=False, continue_train=True, netD='basic_spos',
    #           extra=" --pretrained_name horse2zebra_cutspos_ep400 --lambda_NCE 2 --save_latest_freq 5000")

    # exp.test(extra=' --decoding argmax')
    # exp.test(extra=f' --choice_spos {choice}')

    # exp.fid()
    # exp.eval(metric_list, testB=False)
    #

    # exp.train(batch_size=2, n_epochs=50, n_epochs_decay=0, nce_idt=False, continue_train=True,
    #           extra=" --pretrained_name dog2cat_cutspos_ep800 --lambda_NCE 2 --save_latest_freq 5000")
    # exp.test()
    # exp.fid()  # FID:  28.15505011495911
    # exp.eval(metric_list, testB=False)


def exp_cut_spos_cat2dog():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'vsi', 'total_variation']
    # metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutspos", name="cutspos_ep150", load_size=256, netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=" --ngf 64 --n_blocks 9 --n_choices 4")
    # exp.train(batch_size=2, n_epochs=150, n_epochs_decay=0, nce_idt=False, continue_train=False, netD='basic',
    #           extra=' --lambda_NCE 1 --save_latest_freq 5000')
    exp.eval_arch_searched(eval_num=1)
    choice = str((1, 0, 2, 1, 1, 3, 2, 2, 2)).replace(' ', '')
    # exp.test(extra=f' --choice_spos {choice}')
    # exp.test(extra=' --decoding argmin')
    # exp.test(extra=f' --choice_spos {choice}')
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # fine-tuning
    exp = Experiment(dataset="cat2dog", model="cutspos", name=f"cutspos_ep150_{choice.replace(',', '')}_ep50",
                     load_size=256, netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=f" --choice_spos {choice} --ngf 64 --n_blocks 9 --n_choices 4")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=50, nce_idt=False, continue_train=True, netD='basic',
    #           extra=' --lambda_NCE 1 --save_latest_freq 5000'
    #                 ' --pretrained_name cat2dog_cutspos_ep150')
    # exp.test(extra=f' --choice_spos {choice}')
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1546 sec, ~0.5h
    exp = Experiment(dataset="horse2zebra", model="cutspos", name="cutspos_ep400", load_size=286, netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 5,9,12,16")
    # exp.train(batch_size=2, n_epochs=200, n_epochs_decay=200, nce_idt=False, continue_train=False, netD='basic_spos',
    #           extra=' --lambda_NCE 2 --save_latest_freq 5000')
    choice = str((2, 1, 2, 3, 3, 2, 0, 1, 1)).replace(' ', '')
    exp = Experiment(dataset="horse2zebra", model="cutspos", name="cutspos_ep400_212332011_ep100", load_size=256,
                     netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=f" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 5,9,12,16 --choice_spos {choice}")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=100, nce_idt=False, continue_train=True, netD='basic_spos',
    #           extra=" --pretrained_name horse2zebra_cutspos_ep400 --lambda_NCE 2 --save_latest_freq 5000")

    # exp.test(extra=' --decoding argmax')
    # exp.test(extra=f' --choice_spos {choice}')

    # exp.fid()
    # exp.eval(metric_list, testB=False)
    #

    # exp.train(batch_size=2, n_epochs=50, n_epochs_decay=0, nce_idt=False, continue_train=True,
    #           extra=" --pretrained_name dog2cat_cutspos_ep800 --lambda_NCE 2 --save_latest_freq 5000")
    # exp.test()
    # exp.fid()  # FID:  28.15505011495911
    # exp.eval(metric_list, testB=False)

    # 1546 sec, ~0.5h
    exp = Experiment(dataset="dog2cat", model="cutspos", name="cutspos_ep800", load_size=286, netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 2,5,9,12,16 --save_latest_freq 5000")
    # exp.train(batch_size=2, n_epochs=400, n_epochs_decay=400, nce_idt=False, continue_train=False,
    #           extra=' --lambda_NCE 2')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1546 sec, ~0.5h
    exp = Experiment(dataset="IXI", model="cutspos", name="cutspos_ep20", load_size=286, netG='spos',
                     input_nc=1, output_nc=1, dataroot="../datasets",
                     extra=f" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 2,5,9,12,16")
    # exp.train(batch_size=2, n_epochs=10, n_epochs_decay=10, nce_idt=False, continue_train=False, netD='basic_spos',
    #           extra=' --lambda_NCE 2 --save_latest_freq 5000')
    # exp.test(num_test=500, extra=' --decoding argmin --epoch latest')
    # # exp.fid(epoch='latest')
    # exp.eval(metric_list, testB=False)

    # 1546 sec, ~0.5h
    exp = Experiment(dataset="dog2cat", model="cutspos", name="cutspos_ep200", load_size=286, netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=f" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 2,5,9,12,16")
    # exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, nce_idt=False, continue_train=False, netD='basic_spos',
    #           extra=' --lambda_NCE 2 --save_latest_freq 5000')
    # exp.test(num_test=5000, extra=' --epoch latest')
    # exp.fid(epoch='latest')
    # exp.eval(metric_list, testB=False)

    choice = str((2, 0, 0, 1, 3, 2, 1, 2, 2)).replace(' ', '')
    exp = Experiment(dataset="dog2cat", model="cutspos", name="cutspos_ep100_200132122_ep50", load_size=286,
                     netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=f" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 2,5,9,12,16 --choice_spos {choice}")
    # exp.train(batch_size=2, n_epochs=50, n_epochs_decay=0, nce_idt=False, continue_train=True,
    #           extra=" --pretrained_name dog2cat_cutspos_ep800 --lambda_NCE 2 --save_latest_freq 5000")
    # exp.test()
    # exp.fid()  # FID:  28.15505011495911
    # exp.eval(metric_list, testB=False)

    show_results(metric_list, rowsA, rowsB)


def exp_cut_spos_dog2cat():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'vsi', 'total_variation']
    # metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    # supernet training
    exp = Experiment(dataset="cat2dog", model="cutspos", name="cutspos_ep300", load_size=256, netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=" --ngf 64 --n_blocks 9 --n_choices 3")
    # exp.train(batch_size=2, n_epochs=300, n_epochs_decay=0, nce_idt=False, continue_train=False, netD='basic_spos',
    #           extra=' --lambda_NCE 1 --save_latest_freq 5000')
    # exp.eval_arch_searched(eval_num=5, log='log_cat2dog')
    choice_G = str((1, 2, 0, 0, 2, 0, 2, 2, 1)).replace(' ', '')
    choice_D = str((1, 0, 2, 2, 2)).replace(' ', '')
    # exp.test(script='test_D', extra=' --netD basic_spos --phase val')
    # exp.test(extra=f' --choice_spos {choice}')
    # exp.test(extra=' --decoding argmin')
    # exp.test(extra=f' --choice_spos {choice}')
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # fine-tuning
    exp = Experiment(dataset="cat2dog", model="cutspos",
                     name=f"cutspos_ep300_{choice_G.replace(',', '')}_{choice_D.replace(',', '')}_ep100", load_size=256,
                     netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=f" --choice_G {choice_G} --choice_D {choice_D} --ngf 64 --n_blocks 9 --n_choices 3 --extra_D False")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=100, nce_idt=False, continue_train=True, netD='basic_spos',
    #           extra=' --lambda_NCE 1 --save_latest_freq 5000'
    #                 ' --pretrained_name cat2dog_cutspos_ep300')
    exp.test(extra=' --netD basic_spos')
    exp.fid()
    # exp.eval(metric_list, testB=False)

    # fine-tuning
    exp = Experiment(dataset="cat2dog", model="cutspos", name=f"cutspos_ep150_{choice_G.replace(',', '')}_ep100",
                     load_size=256, netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=f" --choice_spos {choice_G} --ngf 64 --n_blocks 9 --n_choices 3")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=100, nce_idt=False, continue_train=True, netD='basic_spos',
    #           extra=' --lambda_NCE 1 --save_latest_freq 5000'
    #                 ' --pretrained_name cat2dog_cutspos_ep300')
    # exp.test(extra=f' --choice_spos {choice}')
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1546 sec, ~0.5h
    exp = Experiment(dataset="horse2zebra", model="cutspos", name="cutspos_ep400", load_size=286, netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 5,9,12,16")
    # exp.train(batch_size=2, n_epochs=200, n_epochs_decay=200, nce_idt=False, continue_train=False, netD='basic_spos',
    #           extra=' --lambda_NCE 2 --save_latest_freq 5000')
    choice = str((2, 1, 2, 3, 3, 2, 0, 1, 1)).replace(' ', '')
    exp = Experiment(dataset="horse2zebra", model="cutspos", name="cutspos_ep400_212332011_ep100", load_size=256,
                     netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=f" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 5,9,12,16 --choice_spos {choice}")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=100, nce_idt=False, continue_train=True, netD='basic_spos',
    #           extra=" --pretrained_name horse2zebra_cutspos_ep400 --lambda_NCE 2 --save_latest_freq 5000")

    # exp.test(extra=' --decoding argmax')
    # exp.test(extra=f' --choice_spos {choice}')

    # exp.fid()
    # exp.eval(metric_list, testB=False)
    #

    # exp.train(batch_size=2, n_epochs=50, n_epochs_decay=0, nce_idt=False, continue_train=True,
    #           extra=" --pretrained_name dog2cat_cutspos_ep800 --lambda_NCE 2 --save_latest_freq 5000")
    # exp.test()
    # exp.fid()  # FID:  28.15505011495911
    # exp.eval(metric_list, testB=False)

    # 1546 sec, ~0.5h
    exp = Experiment(dataset="dog2cat", model="cutspos", name="cutspos_ep800", load_size=286, netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 2,5,9,12,16 --save_latest_freq 5000")
    # exp.train(batch_size=2, n_epochs=400, n_epochs_decay=400, nce_idt=False, continue_train=False,
    #           extra=' --lambda_NCE 2')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1546 sec, ~0.5h
    exp = Experiment(dataset="IXI", model="cutspos", name="cutspos_ep20", load_size=286, netG='spos',
                     input_nc=1, output_nc=1, dataroot="../datasets",
                     extra=f" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 2,5,9,12,16")
    # exp.train(batch_size=2, n_epochs=10, n_epochs_decay=10, nce_idt=False, continue_train=False, netD='basic_spos',
    #           extra=' --lambda_NCE 2 --save_latest_freq 5000')
    # exp.test(num_test=500, extra=' --decoding argmin --epoch latest')
    # # exp.fid(epoch='latest')
    # exp.eval(metric_list, testB=False)

    # 1546 sec, ~0.5h
    exp = Experiment(dataset="dog2cat", model="cutspos", name="cutspos_ep200", load_size=286, netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=f" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 2,5,9,12,16")
    # exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, nce_idt=False, continue_train=False, netD='basic_spos',
    #           extra=' --lambda_NCE 2 --save_latest_freq 5000')
    # exp.test(num_test=5000, extra=' --epoch latest')
    # exp.fid(epoch='latest')
    # exp.eval(metric_list, testB=False)

    choice = str((2, 0, 0, 1, 3, 2, 1, 2, 2)).replace(' ', '')
    exp = Experiment(dataset="dog2cat", model="cutspos", name="cutspos_ep100_200132122_ep50", load_size=286,
                     netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=f" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 2,5,9,12,16 --choice_spos {choice}")
    # exp.train(batch_size=2, n_epochs=50, n_epochs_decay=0, nce_idt=False, continue_train=True,
    #           extra=" --pretrained_name dog2cat_cutspos_ep800 --lambda_NCE 2 --save_latest_freq 5000")
    # exp.test()
    # exp.fid()  # FID:  28.15505011495911
    # exp.eval(metric_list, testB=False)

    # show_results(metric_list, rowsA, rowsB)


def exp_sim_cut_spos_cat2dog():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'vsi', 'total_variation']
    # metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    # supernet training
    choice = str([3 for _ in range(9)]).replace(' ', '')
    exp = Experiment(dataset="cat2dog", model="simcutspos", name="simcutspos_shuffletf_ep200", load_size=256,
                     netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=f" --ngf 64 --n_blocks 9 --n_choices 4 --choice_spos {choice}")
    # exp.train(batch_size=4, n_epochs=100, n_epochs_decay=100, nce_idt=False, continue_train=False, netD='basic',
    # extra=' --lambda_NCE 1 --save_latest_freq 5000')
    exp.test()
    exp.fid()
    # exp.eval_arch_searched(eval_num=1)
    # choice = str((1, 0, 2, 1, 1, 3, 2, 2, 2)).replace(' ', '')
    # exp.test(extra=f' --choice_spos {choice}')
    # exp.test(extra=' --decoding argmin')
    # exp.test(extra=f' --choice_spos {choice}')


def exp_cyclegan_spos_cat2dog():
    metric_list = ['psnr']
    # metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    # supernet training
    choice = str([3 for _ in range(9)]).replace(' ', '')
    exp = CycleGAN(dataset="horse2zebra", model="cycle_gan_spos", name="cyclespos_ep100", load_size=286, netG='spos',
                   input_nc=3, output_nc=3, dataroot="../datasets",
                   extra=f" --ngf 64 --n_blocks 9 --n_choices 3")
    # exp.train(batch_size=1, n_epochs=100, n_epochs_decay=0, continue_train=False, lambda_identity=0.5,
    #           extra=' --netD basic --save_latest_freq 5000 --display_freq 100 --display_ncols 4')
    # exp.test()

    choice_G_A = str((0, 0, 1, 2, 0, 0, 0, 2, 2)).replace(' ', '')
    choice_G_B = str((0, 2, 1, 0, 0, 0, 0, 2, 2)).replace(' ', '')

    # exp.test(extra=f' --choice_G_A {choice_G_A}')
    # exp.fid()
    # exp.eval_arch_searched(eval_num=10, log='log/horse2zebra_cyclespos')
    # exp.eval_arch_searched_dists(eval_num=10, log='log/horse2zebra_cyclespos')
    # choice = str((1, 0, 2, 1, 1, 3, 2, 2, 2)).replace(' ', '')
    # exp.test(extra=f' --choice_spos {choice}')
    # exp.test(extra=' --decoding argmin')
    # exp.test(extra=f' --choice_spos {choice}')

    # exp.eval_with_return(metric_list, testB=False)

    exp = CycleGAN(dataset="horse2zebra", model="cycle_gan_spos", name=f"cutspos_ep100_{choice_G_A[1:-1]}_ep100",
                   load_size=286, netG='spos',
                   input_nc=3, output_nc=3, dataroot="../datasets",
                   extra=f" --choice_G_A {choice_G_A} --choice_G_B {choice_G_B} --ngf 64 --n_blocks 9 --n_choices 3")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=100, lambda_identity=0.5, continue_train=True,
    #           extra=' --lambda_A 5 --lambda_B 5 --netD basic'
    #                 ' --save_latest_freq 5000  --display_freq 100 --display_ncols 4 --pretrained_name horse2zebra_cyclespos_ep100')
    # exp.test(extra=f' --choice_spos {choice}')
    # exp.eval(metric_list, testB=False)
    exp.test()
    exp.fid()
    # exp.eval(metric_list, testB=False)


def exp_IXI_cut_spos():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'vsi', 'total_variation']
    # metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'haarpsi', 'vsi', 'mdsi', 'gmsd', 'multi_scale_gmsd']
    # metric_list += ['dists', 'lpips', 'pieapp']
    # metric_list += ['mind', 'mutual_info']
    # metric_list += ['total_variation', 'brisque']

    # 1546 sec, ~0.5h
    exp = Experiment(dataset="IXI", model="cutspos", name="cutspos_ep10", load_size=286, netG='spos',
                     input_nc=1, output_nc=1, dataroot="../datasets",
                     extra=" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 5,9,12,16")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False, netD='basic',
    #           extra=' --lambda_NCE 2 --save_latest_freq 5000')
    # exp.test(extra=' --decoding argmin')
    # exp.test(extra=f' --choice_spos {choice}')

    # exp.fid()
    # exp.eval(metric_list, testB=False)

    exp = Experiment(dataset="IXI", model="cutspos", name="cutspos_no_resize_ep10", load_size=256, netG='spos',
                     input_nc=1, output_nc=1, dataroot="../datasets",
                     extra=" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 5,9,12,16")
    # exp.train(batch_size=2, n_epochs=5, n_epochs_decay=5, nce_idt=False, continue_train=False, netD='basic',
    #           extra=' --lambda_NCE 2 --save_latest_freq 5000')
    # # exp.test(extra=' --decoding argmin')
    # choice = str((3, 1, 0, 0, 3, 2, 2, 0, 0)).replace(' ', '')
    # exp.test(extra=f' --choice_spos {choice}')

    # exp.fid()
    # exp.eval(metric_list, testB=False)

    ## supernet training
    exp = Experiment(dataset="IXI", model="cutspos", name="cutspos_ep30", load_size=256, netG='spos',
                     input_nc=1, output_nc=1, dataroot="../datasets",
                     extra=" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 2,5,9,12,16")
    # exp.train(batch_size=2, n_epochs=15, n_epochs_decay=15, nce_idt=False, continue_train=False, netD='basic',
    #           extra=' --lambda_NCE 2 --save_latest_freq 5000')
    # exp.test(extra=' --decoding argmin')
    # exp.eval(metric_list, testB=False)

    ##
    choice = str((1, 2, 3, 2, 3, 0, 1, 0, 1)).replace(' ', '')
    # exp.test(extra=f' --choice_spos {choice}')
    # exp.eval(metric_list, testB=False)

    # fine-tuning
    exp = Experiment(dataset="IXI", model="cutspos", name="cutspos_ep30_123230101_ep10", load_size=256, netG='spos',
                     input_nc=1, output_nc=1, dataroot="../datasets",
                     extra=f" --choice_spos {choice} --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 2,5,9,12,16")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=10, nce_idt=False, continue_train=True, netD='basic',
    #           extra=' --lambda_NCE 1 --save_latest_freq 5000 --pretrained_name IXI_cutspos_ep30')
    # exp.test(extra=f' --choice_spos {choice}')
    # exp.eval(metric_list, testB=False)

    # fine-tuning
    exp = Experiment(dataset="IXI", model="cutspos", name="cutspos_ssim_ep30_123230101_ep10", load_size=256,
                     netG='spos',
                     input_nc=1, output_nc=1, dataroot="../datasets",
                     extra=f" --choice_spos {choice} --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 2,5,9,12,16")
    exp.train(batch_size=2, n_epochs=0, n_epochs_decay=10, nce_idt=False, continue_train=True, netD='basic',
              extra=' --structure_loss ssim --lambda_STR 1 --lambda_NCE 1 --save_latest_freq 5000'
                    ' --pretrained_name IXI_cutspos_ep30')
    exp.test(extra=f' --choice_spos {choice}')
    exp.eval(metric_list, testB=False)

    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1546 sec, ~0.5h
    exp = Experiment(dataset="horse2zebra", model="cutspos", name="cutspos_ep400", load_size=286, netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 5,9,12,16")
    # exp.train(batch_size=2, n_epochs=200, n_epochs_decay=200, nce_idt=False, continue_train=False, netD='basic_spos',
    #           extra=' --lambda_NCE 2 --save_latest_freq 5000')
    choice = str((2, 1, 2, 3, 3, 2, 0, 1, 1)).replace(' ', '')
    exp = Experiment(dataset="horse2zebra", model="cutspos", name="cutspos_ep400_212332011_ep100", load_size=256,
                     netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=f" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 5,9,12,16 --choice_spos {choice}")
    # exp.train(batch_size=2, n_epochs=0, n_epochs_decay=100, nce_idt=False, continue_train=True, netD='basic_spos',
    #           extra=" --pretrained_name horse2zebra_cutspos_ep400 --lambda_NCE 2 --save_latest_freq 5000")

    # exp.test(extra=' --decoding argmax')
    # exp.test(extra=f' --choice_spos {choice}')

    # exp.fid()
    # exp.eval(metric_list, testB=False)
    #

    # exp.train(batch_size=2, n_epochs=50, n_epochs_decay=0, nce_idt=False, continue_train=True,
    #           extra=" --pretrained_name dog2cat_cutspos_ep800 --lambda_NCE 2 --save_latest_freq 5000")
    # exp.test()
    # exp.fid()  # FID:  28.15505011495911
    # exp.eval(metric_list, testB=False)

    # 1546 sec, ~0.5h
    exp = Experiment(dataset="dog2cat", model="cutspos", name="cutspos_ep800", load_size=286, netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 2,5,9,12,16 --save_latest_freq 5000")
    # exp.train(batch_size=2, n_epochs=400, n_epochs_decay=400, nce_idt=False, continue_train=False,
    #           extra=' --lambda_NCE 2')
    # exp.test()
    # exp.fid()
    # exp.eval(metric_list, testB=False)

    # 1546 sec, ~0.5h
    exp = Experiment(dataset="IXI", model="cutspos", name="cutspos_ep20", load_size=286, netG='spos',
                     input_nc=1, output_nc=1, dataroot="../datasets",
                     extra=f" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 2,5,9,12,16")
    # exp.train(batch_size=2, n_epochs=10, n_epochs_decay=10, nce_idt=False, continue_train=False, netD='basic_spos',
    #           extra=' --lambda_NCE 2 --save_latest_freq 5000')
    # exp.test(num_test=500, extra=' --decoding argmin --epoch latest')
    # # exp.fid(epoch='latest')
    # exp.eval(metric_list, testB=False)

    # 1546 sec, ~0.5h
    exp = Experiment(dataset="dog2cat", model="cutspos", name="cutspos_ep200", load_size=286, netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=f" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 2,5,9,12,16")
    # exp.train(batch_size=2, n_epochs=100, n_epochs_decay=100, nce_idt=False, continue_train=False, netD='basic_spos',
    #           extra=' --lambda_NCE 2 --save_latest_freq 5000')
    # exp.test(num_test=5000, extra=' --epoch latest')
    # exp.fid(epoch='latest')
    # exp.eval(metric_list, testB=False)

    choice = str((2, 0, 0, 1, 3, 2, 1, 2, 2)).replace(' ', '')
    exp = Experiment(dataset="dog2cat", model="cutspos", name="cutspos_ep100_200132122_ep50", load_size=286,
                     netG='spos',
                     input_nc=3, output_nc=3, dataroot="../datasets",
                     extra=f" --ngf 64 --n_blocks 9 --n_choices 4 --nce_layers 2,5,9,12,16 --choice_spos {choice}")
    # exp.train(batch_size=2, n_epochs=50, n_epochs_decay=0, nce_idt=False, continue_train=True,
    #           extra=" --pretrained_name dog2cat_cutspos_ep800 --lambda_NCE 2 --save_latest_freq 5000")
    # exp.test()
    # exp.fid()  # FID:  28.15505011495911
    # exp.eval(metric_list, testB=False)

    show_results(metric_list, rowsA, rowsB)


def enco_ct2mr():
    metric_list = ['psnr', 'ssim', 'multi_scale_ssim', 'vsi', 'total_variation']

    exp = CycleGAN(dataset="MR2CT_Reg_PNG", model="cycle_gan", name="cycle_gan_warmup_ep400_id1",
                   load_size=256, gpu_ids='0', input_nc=3, output_nc=3,
                   dataroot="/home/yht/Casit/Datasets/ez/datasets", )
    exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, lambda_identity=0.5, continue_train=False,
              extra=' --save_latest_freq 5000 --display_ncols 3 --display_freq 100'
                    ' --warmup_epochs 20')
    exp.test(extra=' --batch_size 8', epoch='latest')
    exp.fid(epoch='latest')  # FID:
    # exp.traverse_fid()


def main():
    # main()
    # baseline()
    exp = Experiment(dataset="cat2dog", model="enco", name="enco_3,7,13,18,24,28_ep200_hid10",
                     load_size=256, gpu_ids='0',
                     input_nc=3, output_nc=3, dataroot="/home/cas/home_ez/Datasets/EnCo")
    exp.train(batch_size=1, n_epochs=100, n_epochs_decay=100, nce_idt=True, continue_train=False,
              extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 3,7,13,18,24,28 --display_freq 100'
                    ' --lambda_IDT 10 --lambda_NCE 2 --netF mlp_sample_with_DAG'
                    ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 10')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()
    # exp.traverse_fid()

    # s2c_cat2dog()
    exp = Experiment(dataset="horse2zebra", model="enco", name="enco_0,3,7,13,18,24,28,31_ttur_warmup_ep400_id1",
                     load_size=286, input_nc=3, output_nc=3, dataroot="/home/cas/home_ez/Datasets/EnCo",
                     gpu_ids='0')
    # exp.train(batch_size=1, n_epochs=200, n_epochs_decay=200, nce_idt=True, continue_train=False,
    #           extra=' --save_latest_freq 5000 --display_ncols 3 --nce_layers 0,3,7,13,18,24,28,31 --display_freq 100'
    #                 ' --lambda_IDT 1 --lambda_NCE 10 --netF mlp_sample_with_DAG --num_patches 256'
    #                 ' --lr_G 5e-5 --lr_F 5e-5 --lr_D 2e-4 --warmup_epochs 10 --stop_idt_epochs 400 --gan_mode lsgan')
    # exp.test(extra=' --batch_size 8')
    # exp.fid()  # FID:  118.1275130682701
    # exp.traverse_fid()

    # s2c_city()
    # baseline_pretrained()
    # exp_cyclegan_spos_cat2dog()
    # baseline_dog2cat()
    # baseline_pretrained()
    # enco_horse()
    # enco_ct2mr()


if __name__ == '__main__':
    rowsA = []
    rowsB = []

    main()
