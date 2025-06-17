# LLMDetector
This repository contains the implementation for the paper "LLMDetector: Large Language Models with Dual-View Contrastive Learning for Time Series Anomaly Detection".

## Requirements
The dependencies can be installed by: 

```pip install -r requirements.txt```

## Data
The datasets can be downloaded in this [link](https://drive.google.com/drive/folders/1ehugseUxqp1o6Xn60woCFxEEvKq4AT0d?usp=sharing). And the files should be put into `datasets/`, where the original data can be located by `datasets/<dataset_name>`. All the datasets are preprocessed into '.npy' format for convenience.


## Usage
To train and evaluate LLMDetector on a dataset, run the following command:
```python main.py  --anormly_ratio <anormly ratio> --alpha <alpha>  --mode <mode> --dataset <data_name>   --input_c <input dimension>  --win_size <window size>  --patch_size <patch size> --step <step>```

The detailed descriptions about the arguments are as following:


| Parameter Name  | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| anomaly_ratio    | Threshold for determining whether a timestamp is anomalous.                   |
| alpha            | Weighting factor for the instance-level loss in the training objective.     |
| mode             | Specifies the execution mode: either `train` or `test`.                    |
| dataset          | Name of the dataset.                                           |
| input_c          | Number of input channels (i.e., the dimensionality of the input data).     |
| win_size         | Length of the sliding window applied to the time series.                   |
| patch_size       | Size of each patch segmented from the window.                              |
| step             | Step size between consecutive windows during sliding window segmentation.  |


For example, dataset WADI can be directly trained by the following command:
```python main.py --anormly_ratio 0.003  --alpha 0.1 --num_epochs 20    --batch_size 64  --mode train --dataset WADI  --data_path WADI --input_c 123 --win_size 60 --patch_size 6 --step 60```
To test:
```python main.py --anormly_ratio 0.003  --alpha 0.1 --num_epochs 20    --batch_size 64  --mode test --dataset WADI  --data_path WADI --input_c 123 --win_size 60 --patch_size 6 --step 60```

The results will be saved in the directory ```./result```. 

For all the scripts we used, please refer to the directory ```./Scripts```. Or Just directly run the .sh file, for example:
```sh Scripts/MSL.sh```