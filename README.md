<!--
 * @Author: Zehui Lin
 * @Date: 2021-01-07 17:17:46
 * @LastEditors: Zehui Lin
 * @LastEditTime: 2021-01-15 20:22:01
 * @Description: file content
-->
# kaggle moa 
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Zehui-Lin/kaggle_moa/pulls)

Compare Neural Network with XGBoost method in [kaggle moa](https://www.kaggle.com/c/lish-moa/).


## Main Result
|Experiment|Log loss|
|--|--|
|Baseline|0.02191|
|Baseline with categories information |0.02213|
|Baseline with categories information (ema)|0.02031|
|XGBoost|0.01671|
## How to use
Neural Network:
* Step 1. Switch the `ex_name` in `init.py`. `ex_name` can be `baseline`, `add_cate_x`, `add_cate_x_ema`.
* Step 2. Adjust the `batch_size` in `init.py` according to your GPU memory. Higher is perferred.
* Step 3. run the main.py by ```python main.py```.

XGBoost:
* Step 1. run the XGBoost by ```python XGBoost.py```.


## File Structure
```
├── data - the data can be downloaded from kaggle websites
│   ├── sample_submission.csv
│   ├── test_features.csv
│   ├── train_features.csv
│   ├── train_targets_nonscored.csv
│   └── train_targets_scored.csv
├── dataset.py - the definition of torch.utils.data.Dataset class
├── epoch_fun.py - train, validate and test functions
├── init.py - the training configuration of neural network
├── main.py - the main class of framework
├── model.py - the definition of the model class
├── Report.md - the report of the experimental result
├── run_train.py - the implement of cross validation and epoch loop
├── utils.py - the utils of your project
└── XGBoost.py - the implement of XGBoost methods

```


## Acknowledgment
We highly appreciate @[YasufumiNakama](https://github.com/YasufumiNakama) for sharing his great kaggle [noteboos](https://www.kaggle.com/yasufuminakama/moa-pytorch-nn-starter). This repo is mainly based on it. Moreover, thank @[fadel](https://github.com/fadel) for his plug-and-play [ema](https://github.com/fadel/pytorch_ema) module and @[FChmiel](https://www.kaggle.com/fchmiel) for his carefully tuned [XGBoost](https://www.kaggle.com/fchmiel/xgboost-baseline-multilabel-classification) model.


## Contributing
Any kind of enhancement or contribution is welcomed.

## License
The code is licensed with the [MIT](LICENSE) license.