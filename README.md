## Domain Adaption - Pytorch
A Pytorch implement for source-only & [Grad Reverse layer](https://arxiv.org/pdf/1409.7495.pdf) & [MCD](https://arxiv.org/pdf/1712.02560.pdf) &[MDSA](https://arxiv.org/pdf/1812.01754.pdf)
## Usage 
First you need to put the dataset in "dataset" folder.  
For source only :
- python source_only.py $1 $2
  - $1 : source domain 
  - $2 : target domain

```
python source_only.py svhn mnist
```
for Grad Reverse :
- python reverse_grad.py $1 $2
  - $1 : source domain
  - $2 : target domain
```
python reverse_grad svhn mnist
```
for MCD :  
- python mcd.py $1 $2
  - $1 : source domain
  - $2 : target domain
```
python mcd.py svhn mnist
```
for MSDA :  
- python train.py $1 $2 $3 $4 .....
  - $1, $2, $3 : source domain
  - $4 : target domain
(last one will be target domain, others will be source domain.)
```
 python train.py mnist svhn ministm usps
```

## Result

|              | SVHN  --> MNIST | MNISTM  --> MNIST  | USPS --> MNIST |
| :----------: | :-------------: | :----------------: | :------------: |
| source only  |       62%       |        98.5%       |      64.0%     |
| reverse grad |       75%       |        98.8%       |      88.0%     |
|    MCD DA    |       95%       |        99.2%       |      96.5%     |

|               | SVHN+MNISTM+USPS-->MNIST | SVHN+MNISTM+MNIST-->USPS | 
| :-----------: | :----------------------: | :----------------------: |
|     MSDSA     |           98.5%          |           88%            |
 
