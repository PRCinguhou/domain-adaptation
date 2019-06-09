## Domain Adaption - Pytorch
A Pytorch implement for source-only & [Grad Reverse layer](https://arxiv.org/pdf/1409.7495.pdf)

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

## Result

|              | SVHN (Source) --> MNIST(Target) | MNISTM (Source) --> MNIST (Target) |
| :----------: | :-----------------------------: | :--------------------------------: |
| source only  |        62% (30 Epoch)           |             98.5(30 Epoch)         |
| reverse grad |        75% (30 Epoch)           |             98.8(30 Epoch)         |
 
