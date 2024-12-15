![LOGO](https://github.com/sunnyshijuncheng/blob/main/asset/logo.jpg)

Reproducible material for **Meta learning for improved neural network wavefield solutions - Shijun Cheng and Tariq Alkhalifah**

# Project structure
This repository is organized as follows:

* :open_file_folder: **metapinn**: python library containing routines for Meta-PINN;
* :open_file_folder: **asset**: folder containing logo;
* :open_file_folder: **data**: folder to store dataset.

## Supplementary files
To ensure reproducibility, we provide the the data set for meta-train and meta-test stages, and the meta-trained model for Meta-PINN. 

* **Meta-training data set**
Download the meta-training data set [here](https://drive.google.com/drive/folders/1iiZJsiHI3m1jlrHkTXO-YNwvGn3b44Xo?usp=sharing). Then, extract the contents to `dataset/metatrain/`.

* **Meta-testing data set**
Download the meta-testing data set [here](https://drive.google.com/drive/folders/1mUBuahYQbJlDRTcJFLZg-EZTJ0PUZnbm?usp=sharing). Then, extract the contents to `dataset/metatest/`.

* **Meta-initialization model**
Download the meta-initialization neural network model [here](https://drive.google.com/file/d/1GZeMTAHxzTjQFdV27jkhkaRr2rUB1IDM/view?usp=sharing). Then, extract the contents to `/checkpoints/metatrain/`.

## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. 

Remember to always activate the environment by typing:
```
conda activate metapinn
```

## Running code :page_facing_up:
When you have downloaded the supplementary files and have installed the environment, you can run the meta-training and meta-testing demo. 

For meta-train, you can directly run:
```
python metatrain.py
```

For meta-test, you can directly run:
```
python metatest.py
```
**Note:** When you run demo for meta-test, you need open the `metapinn/metatest.py` file to specify the path for meta initialization model. Here, we have provided a meta-initialization model in supplementary file, you can directly load meta-initialization model to perform meta-test.

If you need to compare with a randomly initialized network, you can set the configuration value of `args.use_meta` in the `metapinn/metatest.py` file to `False`,
and then run:
```
python metatest.py
```

**Note:** We emphasize that the training logs for meta-train and meta-test are saved in the `runs/metatrain` and `runs/metatest` file folder, respectively. You can use the `tensorboard --logdir=./` or extract the log to view the changes of the metrics as a function of epoch.

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce A100 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU. Due to the high memory consumption during the meta training phase, if your graphics card does not support large batch task training, please reduce the configuration value of args (`args.update_step`) in the `metapinn/metatrain.py` file.

## Cite us 
```bibtex
@article{cheng2024metapinn,
  title={Meta learning for improved neural network wavefield solutions},
  author={Cheng, Shijun and Alkhalifah, Tariq},
  journal={Surveys in Geophysics},
  pages={1--18},
  year={2024},
  publisher={Springer}
}

