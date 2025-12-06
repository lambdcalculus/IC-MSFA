This repository contains a basic framework for working with MSFAs (MultiSpectral Filter Arrays) for snapshop multispectral imaging
in PyTorch, as well as well as the model studied and used in an undergraduate research project relating to the topic.

The datasets used can be found [here](https://drive.google.com/file/d/1qJNI_BLuiSz1-svzniOmyjRm1RzbrANo/view?usp=sharing).
The training script expects them to be placed them at `./data/datasets`. To train, run:
```
python train.py -c PATH_TO_CONFIG
```

The models used for the results shown in FAPESP's scientific report can be found [here](https://drive.google.com/file/d/1Q__X54m0X-TUsJWYVR4G5D9EIuR7KBde/view?usp=sharing). To reproduce the evaluation, run:
```
python evaluate.py --model PATH_TO_MODEL_DIRECTORY
```
