import torch
# from .cutv0_model import CUTV0Model as CUTModel
from .encov3_model import EnCoV3Model as EnCoModel


class SinEnCoModel(EnCoModel):
    """ This class implements the single image translation model (Fig 9) of
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = EnCoModel.modify_commandline_options(parser, is_train)
        parser.add_argument('--lambda_R1', type=float, default=1.0,
                            help='weight for the R1 gradient penalty')
        parser.add_argument('--lambda_identity', type=float, default=1.0,
                            help='the "identity preservation loss"')

        parser.set_defaults(nce_includes_all_negatives_from_minibatch=True,
                            dataset_mode="singleimage",
                            netG="stylegan2enco",
                            stylegan2_G_num_downsampling=1,
                            netD="stylegan2",
                            netF="mlp_sample",
                            gan_mode="nonsaturating",
                            num_patches=1,
                            nce_layers="0,2,4,6,8,10",
                            lambda_NCE=4.0,
                            ngf=10,
                            ndf=8,
                            lr=0.002,
                            beta1=0.0,
                            beta2=0.99,
                            load_size=1024,
                            crop_size=64,
                            preprocess="zoom_and_patch",
                            )

        if is_train:
            parser.set_defaults(preprocess="zoom_and_patch",
                                batch_size=16,
                                save_epoch_freq=1,
                                save_latest_freq=20000,
                                n_epochs=8,
                                n_epochs_decay=8,

                                )
        else:
            parser.set_defaults(preprocess="none",  # load the whole image as it is
                                batch_size=1,
                                num_test=1,
                                )

        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.visual_names = [x for x in self.visual_names if 'cam' not in x]
        if not self.isTrain:
            self.visual_names = [x for x in self.visual_names if 'idt' not in x]
        if self.isTrain:
            if opt.lambda_R1 > 0.0:
                self.loss_names += ['D_R1']
            if opt.lambda_identity > 0.0:
                self.loss_names += ['idt']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B),
                              dim=0) if (
                    self.opt.nce_idt and self.opt.isTrain and self.get_epoch() <= self.opt.stop_idt_epochs) else self.real_A

        self.fake, feats = self.netG(self.real, layers=self.nce_layers, CUTv2=True)

        self.fake_B = self.fake[:self.real_A.size(0)]
        self.feats = [x[:self.real_A.size(0)] for x in feats]

        if self.opt.nce_idt:
            if (self.opt.isTrain and self.get_epoch() <= self.opt.stop_idt_epochs):
                self.idt_B = self.fake[self.real_A.size(0):]

            # self.feats_idt = [x[self.real_A.size(0):] for x in feats]

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

    def compute_D_loss(self):
        self.real_B.requires_grad_()
        # GAN_loss_D = super().compute_D_loss()
        """Calculate GAN loss for the discriminator"""
        # fake = self.fake_B.detach()
        fake = self.fake_pool.query(self.fake_B.detach())

        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        pred_real = self.netD(self.real_B)
        if self.opt.gan_mode in ['ralsgan', 'rahinge']:
            self.loss_D_fake = self.criterionGAN((pred_fake, pred_real), False).mean()
            self.loss_D_real = self.criterionGAN((pred_real, pred_fake), True).mean()
        else:
            self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
            self.loss_D_real = self.criterionGAN(pred_real, True).mean()

        # combine loss and calculate gradients
        GAN_loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D_R1 = self.R1_loss(pred_real, self.real_B)
        self.loss_D = GAN_loss_D + self.loss_D_R1
        return self.loss_D

    def compute_G_loss(self):
        CUT_loss_G = self.compute_G_loss()
        self.loss_idt = torch.nn.functional.l1_loss(self.idt_B, self.real_B) * self.opt.lambda_identity
        return CUT_loss_G + self.loss_idt

    def R1_loss(self, real_pred, real_img):
        grad_real, = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True, retain_graph=True)
        grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty * (self.opt.lambda_R1 * 0.5)

    def calculate_BYOL_loss(self, feats, cam):
        n_layers = len(self.nce_layers)
        feat_k, feat_q = feats[:n_layers // 2], feats[n_layers // 2:][::-1]

        if self.opt.stop_gradient:
            feat_k = [x.detach() for x in feat_k]

        # cams = []
        # for feat in feat_q:
        # print(cam.shape, feat.shape)
        # cams.append(torch.nn.functional.interpolate(cam, size=feat.shape[2:], mode='bilinear'))

        # prj = not self.opt.no_predictor
        feat_q_pool, sample_ids = self.netF(feat_q, self.opt.num_patches, None)

        feat_k_pool, _ = self.netF2(feat_k, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        # tmp = []
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            # loss = crit(f_q, f_k)
            loss = self.mse_loss(f_q, f_k)
            total_nce_loss += loss.mean()
            # tmp.append(loss.mean().detach().item())
        # print(tmp)
        return total_nce_loss / n_layers * 2
