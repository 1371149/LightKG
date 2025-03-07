## LightKG
The code implementation of paper "LightKG: Efficient Knowledge-Aware Recommendations with Simplified GNN Architecture".

## Introduction
LightKG is a simplified yet powerful Graph Neural Network (GNN)-based Knowledge Graph-aware Recommender System designed to improve recommendation accuracy and training efficiency, particularly in sparse interaction scenarios.

## Enviroment Requirements
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

## Reproducibility & Example to Run the Codes
 We provide the parameters for LightKG on the ML-1M, Last.FM, Book-Crossing and Amazon-book. If you want to run LightKG, choose your target dataset and run as follows:
* Last.FM dataset
```
python main.py --dataset lastfm
```

* Amazon-book dataset
```
python main.py --dataset Amazon-book
```

* ML-1M dataset
```
python main.py --dataset ml-1m
```

* Book-Crossing dataset
```
python main.py --dataset book-crossing
```

## Tips
1. If you want to test our model, please run "python main.py", which by default runs on the Last.fm dataset.
2. If you want to change dataset or use your own dataset, please make sure that the dataset you used is in the file "dataset", then, change the parameter "dataset" in main.py.
3. We provide the code for other GNN-based KGRSs. Some of these are sourced from [Recbole](https://recbole.io/cn/index.html), while KGRec and DiffKG are obtained from [SSLRec](https://github.com/HKUDS/SSLRec). We extend our gratitude to RecBole and SSLRec for their significant contributions to the field of recommender systems.
4. Since CL-SDKG is not open-source and we have not obtained permission from the original authors, we are unable to release its code publicly. If you need it, you can contact the original authors via email, as we did, to inquire.
