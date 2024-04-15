import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils

import numpy as np
import torch
import torch.nn.functional as F
from IPython import embed
import lpips

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr

#### options
parser = argparse.ArgumentParser()
parser.add_argument("-opt", type=str, required=True, help="Path to options YMAL file.")
opt = option.parse(parser.parse_args().opt, is_train=False)

opt = option.dict_to_nonedict(opt)

#### mkdir and logger
util.mkdirs(
    (
        path
        for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    )
)

os.system("rm ./result")
os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

util.setup_logger(
    "base",
    opt["path"]["log"],
    "test_" + opt["name"],
    level=logging.INFO,
    screen=True,
    tofile=True,
)
logger = logging.getLogger("base")
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt["datasets"].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info(
        "Number of test images in [{:s}]: {:d}".format(
            dataset_opt["name"], len(test_set)
        )
    )
    test_loaders.append(test_loader)

# load pretrained model by default
model = create_model(opt)
device = model.device

sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
sde.set_model(model.model)
lpips_fn = lpips.LPIPS(net='alex').to(device)

scale = opt['degradation']['scale']

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt["name"]  # path opt['']
    logger.info("\nTesting [{:s}]...".format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
    util.mkdir(dataset_dir)

    test_times = []

    for i, test_data in enumerate(test_loader):
        img_path = test_data["LQ_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        #### input dataset_LQ
        LQ = test_data["LQ"]
        H, W = LQ.shape[2:]
        flag = False
        if max(H, W) >= 4000:
            flag = True
            LQ = F.interpolate(LQ, scale_factor=0.8)

        noisy_state = sde.noise_state(LQ)

        model.feed_data(noisy_state, LQ, None)
        tic = time.time()
        model.test(sde, mode='posterior', save_states=False)
        # model.test(sde, mode='sde', save_states=False)
        toc = time.time()
        test_times.append(toc - tic)

        visuals = model.get_current_visuals(need_GT=False)
        SR_img = visuals["Output"]
        if flag: SR_img = F.interpolate(SR_img, size=(H, W))
        output = util.tensor2img(SR_img.squeeze())  # uint8
        
        suffix = opt["suffix"]
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + ".png")
        else:
            save_img_path = os.path.join(dataset_dir, img_name + ".png")
        util.save_img(output, save_img_path)

        logger.info(img_name)


    print(f"average test time: {np.mean(test_times):.4f}")
