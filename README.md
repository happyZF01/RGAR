# RGAR

This repository contains inference code for reliability-gated adaptive
redundancy (RGAR). 


## Weights

The proposed RGAR model is trained for 400k iterations. The checkpoint obtained at 400k iterations under the AWGN channel is used for the experiments in the paper.

https://drive.google.com/file/d/1CkHGfeTdO9Rx38-gV5QCXG-xwzt9DUT2/view?usp=drive_link


## Inference

Install the CUDA Tree-VQ extension and dependencies:

```bash
pip install -r requirements.txt
pip install -e tree_vq_ext
```

AWGN:

```bash
python inference.py --input /path/to/images --output outputs/awgn --checkpoint xxx.ckpt --channel awgn --snrs=-5,0,5
```
