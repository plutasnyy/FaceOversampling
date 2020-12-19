import torch
from psp_models.psp import pSp
from argparse import Namespace
from psp_configs import data_configs
from psp_datasets.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
import os


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

    def fit(self, dataset_path):
        dataset = InferenceDataset(root=dataset_path,
                                   transform=self.transforms_dict['transform_inference'],
                                   opts=self.opts)

        self.dataloader = DataLoader(dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=int(self.opts.test_workers),
                                     drop_last=True)
        self.net.eval()
        self.net.cuda()

    def interpolate(self, img1, img2, alpha):
        with torch.no_grad():
            _, latent_to_inject = self.net(img1.unsqueeze(0).cuda().float(),
                                           return_latents=True)

        input_cuda = img2.unsqueeze(0).cuda().float()
        result_batch, latent = self.net(input_cuda, randomize_noise=False, resize=self.resize_outputs,
                                        return_latents=True, latent_mask=self.latent_mask,
                                        inject_latent=latent_to_inject[0].unsqueeze(0), alpha=alpha)
        return result_batch[0]

    def transform(self):
        batch_results = []
        for input_batch in self.dataloader:
            batch_results.append(self.interpolate(input_batch[0], input_batch[1], 0.5))
        return batch_results
