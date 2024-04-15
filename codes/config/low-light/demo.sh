#!/bin/bash

#############################################################
### training ###

# for single GPU
python train.py -opt=options/train/entropy-refusion.yml

# for multiple GPUs
# torchrun --nproc_per_node 2 -m train -opt=options/train/entropy-refusion.yml

#############################################################

### testing ###
# python test.py -opt=options/test/refusion.yml

#############################################################
