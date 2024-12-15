![LOGO](https://github.com/DeepWave-Kaust/Project-Template/blob/main/asset/logo.png)

Reproducible material for **Meta learning for improved neural network wavefield solutions - Shijun Cheng and Tariq Alkhalifah**

# Project structure
This repository is organized as follows:

* :open_file_folder: **metapinn**: python library containing routines for Meta-PINN;
* :open_file_folder: **asset**: folder containing logo;
* :open_file_folder: **data**: folder to store dataset.

## Supplementary files
To ensure reproducibility, we provide the the data set for meta-train and meta-test stages, and the meta-trained model for Meta-PINN. 

* **Meta-training and Meta-testing data set**
Download the meta-training and meta-testing data set [here](https://kaust.sharepoint.com/sites/M365_Deepwave_Documents/Shared%20Documents/Forms/AllItems.aspx?csf=1&web=1&e=BiJZyw&CID=91cc5ce7%2D0cbb%2D4cd4%2D8e52%2Ddf708ab9d986&FolderCTID=0x0120009F9BE65BA42D194BBEFB62CBD730AF6A&id=%2Fsites%2FM365%5FDeepwave%5FDocuments%2FShared%20Documents%2FRestricted%20Area%2FREPORTS%2FDW0062%2Fdata). Then, extract the contents to `dataset/metatrain/` and `dataset/metatest/`, respectively.

* **Meta-initialization model**
Download the meta-initialization neural network model [here](https://kaust.sharepoint.com/sites/M365_Deepwave_Documents/Shared%20Documents/Forms/AllItems.aspx?csf=1&web=1&e=BiJZyw&CID=91cc5ce7%2D0cbb%2D4cd4%2D8e52%2Ddf708ab9d986&FolderCTID=0x0120009F9BE65BA42D194BBEFB62CBD730AF6A&id=%2Fsites%2FM365%5FDeepwave%5FDocuments%2FShared%20Documents%2FRestricted%20Area%2FREPORTS%2FDW0062%2Fdata%2Fmeta%5Ftrained%2Epth&parent=%2Fsites%2FM365%5FDeepwave%5FDocuments%2FShared%20Documents%2FRestricted%20Area%2FREPORTS%2FDW0062%2Fdata). Then, extract the contents to `/checkpoints/metatrain/`.

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
**Note:** When you run demo for meta-test, you need open the `metalrpinn/meta_test.py` file to specify the path for meta initialization model. Here, we have provided a meta-initialization model in supplementary file, you can directly load meta-initialization model to perform meta-test.

If you need to compare with a randomly initialized network, you can comment out lines 52 in the `metalrpinn/meta_test.py` file as follows
```
# meta.load_state_dict(torch.load(dir_meta, map_location=device))
```
and then run:
```
sh run_metatest.sh
```

**Note:** We emphasize that the training logs for meta-train and meta-test are saved in the `runs/metatrain` and `runs/metatest` file folder, respectively. You can use the `tensorboard --logdir=./` or extract the log to view the changes of the metrics as a function of epoch.

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce A100 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU. Due to the high memory consumption during the meta training phase, if your graphics card does not support large batch task training, please reduce the configuration value of args (`args.ntask`) in the `metalrpinn/meta_train.py` file.

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX 3090 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.

## Cite us 
DWXXX - Author1 et al. (2022) Report title.

