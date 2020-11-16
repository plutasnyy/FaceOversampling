# Adapted from scripts/inference.py
# uruchmialem to z
# --exp_dir . --checkpoint_path models/pSp_weights/psp_ffhq_encode.pt --data_path oversampling/example_images --latent_mask=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17
import os
from argparse import Namespace

import numpy as np
import torch
from PIL.Image import fromarray
from torch.utils.data import DataLoader
from tqdm import tqdm

from psp_configs import data_configs
from psp_datasets.inference_dataset import InferenceDataset
from psp_models.psp import pSp
from psp_options.test_options import TestOptions
from psp_utils.common import tensor2im


def run():
    test_opts = TestOptions().parse()

    if test_opts.resize_factors is not None:
        assert len(
            test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
                                        'downsampling_{}'.format(test_opts.resize_factors))
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
                                        'downsampling_{}'.format(test_opts.resize_factors))
    else:
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test psp_options with psp_options used during psp_training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    # opts['device']='cpu'
    opts = Namespace(**opts)

    print("Loaded options")

    net: pSp = pSp(opts)

    print("Loaded network")

    net.eval()
    net.cuda()
    print("Network in CUDA")

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    for input_batch in tqdm(dataloader):
        with torch.no_grad():
            # input_cuda = input_batch.float()
            input_cuda = input_batch.cuda().float()
            result_batch, latent = net(input_cuda, randomize_noise=False, resize=opts.resize_outputs,
                                       return_latents=True)

    print(latent.shape)
    latent_mask = [int(l) for l in opts.latent_mask.split(",")]
    with torch.no_grad():
        res = net(latent[0], input_code=True, randomize_noise=False, resize=opts.resize_outputs, alpha=0.5,
                  inject_latent=latent[1], latent_mask=latent_mask)

    result = tensor2im(res[0])
    fromarray(np.array(result)).save('.')


if __name__ == '__main__':
    run()
