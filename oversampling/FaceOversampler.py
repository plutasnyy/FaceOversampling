import os
from argparse import Namespace
from collections import defaultdict
from random import sample, uniform

import imagehash
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image

from psp_configs import data_configs
from psp_models.psp import pSp
from psp_utils.common import tensor2im


class FaceOversampler(object):
    def __init__(self, latent_mask, exp_dir, checkpoint_path, resize_factors=None):
        if resize_factors is not None:
            assert len(
                resize_factors.split(
                    ',')) == 1, "When running inference, provide a single downsampling factor!"
            out_path_results = os.path.join(exp_dir, 'inference_results',
                                            'downsampling_{}'.format(resize_factors))
            out_path_coupled = os.path.join(exp_dir, 'inference_coupled',
                                            'downsampling_{}'.format(resize_factors))
        else:
            out_path_results = os.path.join(exp_dir, 'inference_results')
            out_path_coupled = os.path.join(exp_dir, 'inference_coupled')

        os.makedirs(out_path_results, exist_ok=True)
        os.makedirs(out_path_coupled, exist_ok=True)

        ckpt = torch.load(checkpoint_path, map_location='cpu')
        opts = ckpt['opts']
        opts.update({'exp_dir': exp_dir, 'checkpoint_path': checkpoint_path, 'resize_factors': resize_factors})
        if 'learn_in_w' not in opts:
            opts['learn_in_w'] = False
        opts = Namespace(**opts)

        self.net: pSp = pSp(opts)
        self.opts = opts
        self.latent_mask = [int(l) for l in latent_mask.split(",")]
        self.resize_outputs = False
        self.transforms_dict = data_configs.DATASETS[opts.dataset_type]['transforms'](opts).get_transforms()
        self.dataloader = None
        self.batch_size = 2

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.net.eval()
        self.net.cuda()

    def interpolate(self, img1, img2, alpha):
        with torch.no_grad():
            _, latent_to_inject = self.net(img1.unsqueeze(0).cuda().float(),
                                           return_latents=True)

        input_cuda = img2.unsqueeze(0).cuda().float()
        result_batch, latent = self.net(input_cuda, resize=self.resize_outputs,
                                        return_latents=True, latent_mask=self.latent_mask,
                                        inject_latent=latent_to_inject[0].unsqueeze(0), alpha=alpha)
        return result_batch[0]

    def inject_random_face(self, img, alpha, quantity=1):
        multi_modal_outputs = []
        for vec_to_inject in range(quantity):
            cur_vec = torch.from_numpy(np.random.randn(1, 512).astype('float32')).unsqueeze(0).to("cuda")
            _, latent_to_inject = self.net(cur_vec, input_code=True, return_latents=True)
            res = self.net(img.unsqueeze(0).cuda().float(), latent_mask=self.latent_mask,
                           inject_latent=latent_to_inject, alpha=alpha)
            multi_modal_outputs.append(res[0])
        return multi_modal_outputs

    def fit_transform(self, dataset_path, result_path):
        if not os.path.exists(result_path + '/new_imgs'):
            os.mkdir(result_path + '/new_imgs')

        df = pd.read_csv(dataset_path)
        imgs = defaultdict(list)

        for i, g in df.groupby("age"):
            for idx, im in g.iterrows():
                imgs[im['age']].append(im['aligned_path'])

        no_samples = len(df) / len(imgs)
        new_imgs = list()

        for i in imgs.keys():
            for j in range(int(no_samples - len(imgs[i]))):
                paths = sample(imgs[i], 2)
                pics = list()
                for p in paths:
                    image = Image.open(p)
                    pics.append(self.transform(image))
                alpha = uniform(0, 1)
                new_img = tensor2im(self.interpolate(pics[0], pics[1], alpha))
                img_hash = str(imagehash.average_hash(new_img)) + ".jpg"
                new_img.save(result_path + '/new_imgs/' + img_hash)
                new_imgs.append({'aligned_path': result_path + '/new_imgs/' + img_hash, 'age': i})

        imgs_df = pd.DataFrame(new_imgs)
        imgs_df.to_csv(result_path + "/train_new.csv")
        print("Done")
