## Macro-expression spotting
This is our implementation for macro-expression spotting in MEGC 2021. We employ the commonly used "CNN+RNN" framework, and a transfer learning approach is adopted to alleviate the overfitting problem.
### Requirements
- Python 3.7
- PyTorch 1.4.0
- Pandas
- Matplotlib
- NumPy
### Train
- train on CASME
```
python train.py -c config_casme.config
```
- train on SAMM
```
python train.py -c config_samm.config
```
Note that modify the data path in the configuration file to yours. For the pre-trained EfficientFace model, we refer you to the [official implementation](https://github.com/zengqunzhao/EfficientFace) and you need to train the model on RAF-DB by yourself.
### Evaluation
- evaluate on CASME
```
python evaluate_casme.py --model_dir [directory of your pre-trained model]
```
- evaluate on SAMM
```
python evaluate_casme.py --model_dir [directory of your pre-trained model]
```
### Acknowledgement
Our implementation is based on the awesome [pytorch template](https://github.com/victoresque/pytorch-template)

