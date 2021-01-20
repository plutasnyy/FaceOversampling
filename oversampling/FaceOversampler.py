import os
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from random import sample, uniform

import imagehash
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from psp_configs import data_configs
from psp_models.psp import pSp
from psp_utils.common import tensor2im


class FaceOversampler(object):
    def __init__(self, exp_dir, checkpoint_path, resize_factors=None, latent_mask=None, latent_mask_mix=None):
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

        if latent_mask is None:
            self.latent_mask_interpolate = [i for i in range(18)]
        else:
            self.latent_mask_interpolate = latent_mask

        if latent_mask_mix is None:
            self.latent_mask_mix = [i for i in range(9, 18)]
        else:
            self.latent_mask_mix = latent_mask_mix

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
                                        return_latents=True, latent_mask=self.latent_mask_interpolate,
                                        inject_latent=latent_to_inject[0].unsqueeze(0), alpha=alpha)
        interpolated_img = tensor2im(result_batch[0]).resize((256, 256))
        mixed = self.inject_random_face(self.transform(interpolated_img), alpha=uniform(0.5, 0.8), quantity=1)
        return mixed[0]

    def inject_random_face(self, img, alpha, quantity=1):
        multi_modal_outputs = []
        torch.cuda.empty_cache()

        with torch.no_grad():
            for vec_to_inject in range(quantity):
                cur_vec = torch.from_numpy(np.random.randn(1, 512).astype('float32')).cuda()
                _, latent_to_inject = self.net(cur_vec, input_code=True, return_latents=True)
                res = self.net(img.unsqueeze(0).cuda().float(), latent_mask=self.latent_mask_mix,
                               inject_latent=latent_to_inject, alpha=alpha)
                multi_modal_outputs.append(tensor2im(res[0]))
        torch.cuda.empty_cache()

        return multi_modal_outputs

    def fit_transform(self, dataset_path, result_path):
        Path(result_path + '/images').mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(dataset_path)
        imgs = defaultdict(list)

        for i, g in df.groupby("age"):
            for idx, im in g.iterrows():
                imgs[im['age']].append(im['aligned_path'])

        no_samples = len(df) / len(imgs)

        new_imgs = list()

        for i in tqdm(imgs.keys(), total=len(imgs)):
            for j in range(int(no_samples - len(imgs[i]))):
                paths = sample(imgs[i], 2)
                pics = list()
                for p in paths:
                    image = Image.open(p)
                    pics.append(self.transform(image))
                alpha = uniform(0, 1)
                new_img = self.interpolate(pics[0], pics[1], alpha)

                img_hash = str(imagehash.average_hash(new_img)) + ".jpg"
                new_img.save(result_path + '/images/' + img_hash)
                new_imgs.append({'aligned_path': '/images/' + img_hash, 'age': i, 'base_path': str(paths[0])})

        imgs_df = pd.DataFrame(new_imgs)
        imgs_df.to_csv(result_path + "/train.csv")
        print("Done")
