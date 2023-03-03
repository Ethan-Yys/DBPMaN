# DBPMaN

tensorflow：1.4.1

python: 2.7

## Getting Started
1. download and sample data:

download UserBehavior.csv from https://tianchi.aliyun.com/dataset/649
head -20000000 UserBehavior.csv > train_data.csv

2. data processing:

python3 process-py36.py

3. training:

python  script/train.py train DBPMaN
