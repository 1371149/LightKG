Official implement of paper "LightKG: Advancing Knowledge-Aware Recommendations with Simplified GNN Architecture", SIGIR 2025.

## Requirements
Tested on python 3.9 and Ubuntu 20.04.
1. [pytorch](https://pytorch.org/)==2.0.0
2. [recbole](https://recbole.io/)==1.1.1
3. [lightgbm](https://github.com/microsoft/LightGBM/tree/master/python-package)
4. [xgboost](https://github.com/dmlc/xgboost)
5. [ray](https://www.ray.io/)
6. [thop](https://github.com/Lyken17/pytorch-OpCounter)
7. [torch_scatter](https://github.com/rusty1s/pytorch_scatter/tree/master)

## Dataset process
You can simply use the processed dataset provided by us. If you want to process the dataset yourself, please refer to [recbole](https://recbole.io/) and our paper for details.

## Tips
1. If you want to test our model, please run "python launch.py".
2. By default, we run the LightKG on the ML-1M dataset. If you want to change dataset or use your own dataset, please make sure that the dataset you used is in the file "dataset", then, change the parameter "dataset" in launch.py.
3.  We provide the parameters for LightKG on the ML-1M, Last.FM, Book-Crossing and Amazon-book. If you want to use them, please write the corresponding yaml file into the "config_file_list" in the "launch.py".
