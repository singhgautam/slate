
# SLATE

This is the official source code for _SLATE_. We provide the code for the model, the training code and a dataset loader for the 3D Shapes dataset. This code is implemented in Pytorch.

Arxiv: https://arxiv.org/pdf/2110.11405.pdf <br>
Project Page: https://sites.google.com/view/slate-autoencoder

### Dataset
The current release provides a boilerplate code to train the model on the 3D Shapes dataset. The dataset class is provided in `shapes_3d.py`. You can edit or replace this class if you need to run the code on a different dataset. The 3D Shapes dataset can be downloaded from the official URL https://console.cloud.google.com/storage/browser/3d-shapes. This should produce a dataset file `3dshapes.h5`. During training, the path to this dataset file needs to be provided using the argument `--data_path`.

### Training
To train the model, simply execute:
```bash
python train.py
```
Check `train.py` to see the full list of training arguments.

### Outputs
The training code produces Tensorboard logs. To see these logs, run Tensorboard on the logging directory that was provided in the training argument `--log_path`. These logs contain the training loss curves and visualizations of reconstructions and object attention maps.


### Hyperparameters of Interest
- **Learning Rate** can be tuned using the training argument `--lr_main` and different choices can affect the characteristics of the object attention maps. Values 3e-4 and 1e-4 can be considered as good guesses when trying different datasets.
- **Number of Slots** can be tuned using the training argument `--num_slots`. Number of slots should be set higher than the number of objects you expect to see in the images.
- **Number of Slot Attention Iterations** can be tuned using the training argument `--num_iterations`. In general, avoid setting this to a large value because too many iterations can prevent slots from learning to diversify and attach to different objects. At the same time, having too few iterations may not be enough for attention maps to refine properly. Some values that are worth trying are 3 and 7 when trying different datasets.

### Code Files
This repository provides the following files.
- `train.py` contains the main code for running the training.
- `slate.py` provides the model class for SLATE.
- `shapes_3d.py` contains the dataset class for 3D Shapes dataset.
- `dvae.py` provides the encoder and the decoder for Discrete VAE.
- `slot_attn.py` provides the model class for Slot Attention encoder.
- `transformer.py` provides the model classes for Transformer.
- `utils.py` provides helper classes and functions for the implementation.

### Citation
```
@misc{singh2021illiterate,
      title={Illiterate DALL-E Learns to Compose}, 
      author={Gautam Singh and Fei Deng and Sungjin Ahn},
      year={2021},
      eprint={2110.11405},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
