# Implementation paper convolutional neural networks for sentence classfication

## Download dataset

```
gdown 171vJ_msiZp9uPLRLUwN2Fq719ZfilPTh
gdown 17BO5HZtpU6JXc982CKTNpeDHWEgYo-DM
```

## Download word2vec pretrained for Vietnamese
```
gdown 1-DiVYL7bIhtvrXotWyZPQt4oNmUgXQDE
```
## Run

```
pip install -r requirements.txt

python run.py\
        --epochs 10\
        --batch_size 32\
        --max_sen_lem 300\
        --lr 0.001

```
