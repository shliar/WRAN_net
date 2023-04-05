# WRAN_net


Requirements

To install requirements:

           pip install -r requirements.txt

Datasets split:
           
           Move the datafile to dataset/
           Run 'python write_dataset_filelist.py'
  
           


Training

To train the feature extractors in the paper, run this command:

           python train.py --dataset [dataset] --method [S2M2_R] --model [WideResNet28_10] --train_aug

Evaluation

To evaluate my model on dataset, run:
For dataset

           python save_plk.py

           python test.py



Hyperparameter setting

common setting:

           1-shot: k=10 kappa=9 beta=0.5 5-shot: k=4 kappa=1 beta=0.75

Contact the author e-mail：1843134669@qq.com
