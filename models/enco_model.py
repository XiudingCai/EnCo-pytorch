import numpy as np
import torch
from .base_model import BaseModel
from . import networks
import util.util as util
from util.image_pool import ImagePool

from itertools import chain
import torch.nn.functional as F
from .patchnce import PatchNCELoss as PatchNCELoss


class EnCoModel(BaseModel):
    """ This class implements EnCo model, described in the paper
    Rethinking the Paradigm of Content Constraints in GAN-based Unpaired Image-to-Image Translation
    Xiuding Cai, Yaoyao Zhu, Dong Miao, Linjie Fu, Yu Yao.
    AAAI, 2024
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_IDT', type=float, default=10.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='3,7,13,18,24,28',
                            help='compute BYOL loss on paired features, e.g. (0,31), (3,28), (7,24), (13,18)')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='cam_mlp_sample',
                            choices=['sample', 'reshape', 'mlp_sample', 'mlp_sample_with_DAG'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.add_argument('--lr_G', type=float, default=5e-5, help='initial learning rate for adam')
        parser.add_argument('--lr_D', type=float, default=2e-4, help='initial learning rate for adam')
        parser.add_argument('--lr_F', type=float, default=5e-5, help='initial learning rate for adam')

        # DAG
        parser.add_argument('--sampling_strategy', type=str, default='DAG', help='random, DAG')
        parser.add_argument('--oversample_ratio', type=int, default=4, help='number of patches per layer')
        parser.add_argument('--random_ratio', type=float, default=.5, help='number of patches per layer')

        # EnCo
        parser.add_argument('--stop_idt_epochs', type=int, default=-1,
                            help='stop the use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--stop_gradient', type=util.str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--two_F', type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--no_predictor', type=util.str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--decay_idt', type=util.str2bool, nargs='?', const=True, default=False)

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()
        parser.set_defaults(nce_idt=True, lambda_NCE=1.0, pool_size=0)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'BYOL']
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if self.opt.stop_idt_epochs < 0 and self.opt.isTrain:
            self.opt.stop_idt_epochs = self.opt.n_epochs

        if self.isTrain:
            self.epochs_num = self.opt.n_epochs + self.opt.n_epochs_decay

        if (opt.nce_idt and self.opt.isTrain and self.get_epoch() <= self.opt.stop_idt_epochs) or not self.opt.isTrain:
            self.loss_names += ['BYOL_Y']
            self.visual_names += ['idt_B']
        if self.opt.isTrain:
            self.visual_names += ['cam_fake', 'cam_real']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
            if self.opt.two_F:
                self.model_names.append('F2')
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,
                                      opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                      opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        opt.two_F = False
        if opt.two_F:
            self.netF2 = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                           opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        else:
            self.netF2 = self.netF

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                                          opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.fake_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers[:len(self.nce_layers) // 2]:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            # self.criterionIdt = torch.nn.SmoothL1Loss(beta=.5).to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_G, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()  # calculate gradients for D
            self.compute_G_loss().backward()  # calculate graidents for G
            if 'mlp' in self.opt.netF:
                if self.opt.two_F:
                    self.optimizer_F = torch.optim.Adam(chain(self.netF.parameters(), self.netF2.parameters()),
                                                        lr=self.opt.lr_F,
                                                        betas=(self.opt.beta1, self.opt.beta2))
                else:
                    self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr_F,
                                                        betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if 'mlp' in self.opt.netF:
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if 'mlp' in self.opt.netF:
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)

        if self.opt.flip_equivariance:
            if self.opt.isTrain and (np.random.random() < 0.5):
                self.real_A = torch.flip(self.real_A, [3])
            if self.opt.isTrain and (np.random.random() < 0.5):
                self.real_B = torch.flip(self.real_B, [3])

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B),
                              dim=0) if (
                                                self.opt.nce_idt and self.opt.isTrain and self.get_epoch() <= self.opt.stop_idt_epochs) or not self.opt.isTrain else self.real_A

        self.fake, feats = self.netG(self.real, layers=self.nce_layers, CUTv2=True)

        self.fake_B = self.fake[:self.real_A.size(0)]
        self.feats = [x[:self.real_A.size(0)] for x in feats]

        if self.opt.nce_idt:
            if (self.opt.isTrain and self.get_epoch() <= self.opt.stop_idt_epochs) or not self.opt.isTrain:
                self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        # fake = self.fake_B.detach()
        fake = self.fake_pool.query(self.fake_B.detach())

        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        pred_real = self.netD(self.real_B)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        self.loss_D_real = self.criterionGAN(pred_real, True).mean()

        self.cam_fake = torch.nn.functional.interpolate(pred_fake, size=fake.shape[2:], mode='bilinear')
        self.cam_real = torch.nn.functional.interpolate(pred_real, size=fake.shape[2:], mode='bilinear')

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and BYLO loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        # maintain content
        if self.opt.lambda_NCE > 0.0:
            cam = pred_fake.detach()
            self.loss_BYOL = self.calculate_BYOL_loss(self.feats, cam=cam) * self.opt.lambda_NCE
        else:
            self.loss_BYOL, self.loss_BYOL_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0 and self.get_epoch() <= self.opt.stop_idt_epochs:
            if self.opt.decay_idt:
                lambda_IDT = self.opt.lambda_IDT * (self.epochs_num - self.get_epoch()) / self.epochs_num
            else:
                lambda_IDT = self.opt.lambda_IDT
            self.loss_BYOL_Y = self.criterionIdt(self.real_B, self.idt_B) * lambda_IDT
            loss_BYOL_both = (self.loss_BYOL + self.loss_BYOL_Y) * 0.5
        else:
            loss_BYOL_both = self.loss_BYOL
            self.loss_BYOL_Y = 0

        self.loss_G = self.loss_G_GAN + loss_BYOL_both
        return self.loss_G

    def calculate_BYOL_loss(self, feats, cam):
        n_layers = len(self.nce_layers)
        feat_k, feat_q = feats[:n_layers // 2], feats[n_layers // 2:][::-1]

        if self.opt.stop_gradient:
            feat_k = [x.detach() for x in feat_k]

        cams = []
        for feat in feat_q:
            cams.append(torch.nn.functional.interpolate(cam, size=feat.shape[2:], mode='bilinear'))

        prj = not self.opt.no_predictor
        feat_q_pool, sample_ids = self.netF(feat_q, self.opt.num_patches, None, cams=cams, prj=prj)

        feat_k_pool, _ = self.netF2(feat_k, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0

        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = self.mse_loss(f_q, f_k)
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers * 2

    def mse_loss(self, feat_q, feat_k):
        if self.opt.stop_gradient:
            feat_k = feat_k.detach()
        return 2 - 2 * F.cosine_similarity(feat_q, feat_k, dim=-1)
