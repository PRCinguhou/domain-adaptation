# Domain Adaption - Pytorch
A Pytorch implement for source-only & [Grad Reverse layer](https://arxiv.org/pdf/1409.7495.pdf)

# Usage 
First you need to put the dataset in "dataset" folder.
for source only :
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

