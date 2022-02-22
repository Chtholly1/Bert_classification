## Bert分类模型
### 前言
这是一个基于**AlBert(以下Bert若不特别强调都是AlBert)** 模型实现的文本分类模型，笔者把其中一些常用的效果提升的方法都进行了封装，可以一键关闭和打开。
python环境：
pytorch:    1.10
transformers:   4.12.5
numpy:  1.19.0

### 正文
目前已有的方法：
1.embedding对抗训练 FGM/PGD
2.滑动平均 EMA
3.对比学习损失函数 SCL_loss
4.数据增强 mixup

#### 参考论文：
- [Augmenting Data with Mixup for Sentence Classification](https://arxiv.org/pdf/1905.08941.pdf)(数据增强mixup)
- [Supervised Contrastive Learning for Pre-trained Language Model Fine-tuning](https://arxiv.org/pdf/2011.01403.pdf)(SCL_loss)
- [如何优化BERT等预训练模型-Tricks总结](https://jishuin.proginn.com/p/763bfbd54336)

#### 参考代码：
- https://github.com/HobbitLong/SupContrast (图像分类问题的SCL_loss，我是基于这个代码进行修改的)

#### 推荐超参数：
learning_rate=2e-5 
batch_size=16 
max_length=256(不超过512) 
attack_type='FGM' 
use_EMA=True 
use_SCL_loss=False 
use_mixup=False 
lr_decay = 0.5(当前epoch上dev的loss不再下降时生效)

ps:可根据自身任务进行调整

### 训练命令
`python train_adv.py`

读取文件路径，以及模型保存路径等都写在conf/config.py中的，自己看一下就知道了。
数据不方便公开，读者可自行下载分类任务数据。

### 效果对比

| model_name | acc | / | / |
| --- | --- | --- | --- |
| Bert | 87.5 |  |  |
| Bert+FGM | 88.9 |  |  |
| Bert+FGM+EMA | 89.1 |  |  |
| Bert+SCL_loss | 87.1 |  |  |
| Bert+Mixup |87.3 | | |

ps:SCL_LOSS和Mixup方法在本人数据集上都没有起到效果，猜测有两种可能：
1. 这两种方法不适合这个数据集
2. 笔者对论文的细节理解有一些谬误之处，实现并非完全准确，读者可根据论文原文和代码自行进行修改。
