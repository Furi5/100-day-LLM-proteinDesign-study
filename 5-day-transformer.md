# 学习 transformer

:dizzy_face: 这周，真的，每天都有时间摸鱼哇

### :star2:学习目标

:black_square_button: 学习 transformer 模型的数据如何预处理
:black_square_button: 尝试自己画一下 transformer 的框架图来学习，transformer 的架构
:black_square_button: 了解 transformer 架构的好处

### 📒学习笔记

<https://blog.csdn.net/qq_41764621/article/details/126210936>

```python
from tensorboardX import SummaryWriter

'''
`SummaryWriter` 类提供了一个高级 API，用于在给定目录中创建事件文件，
并向其中添加摘要和事件。 该类异步更新文件内容。 这允许训练程序调用方法以直接
从训练循环将数据添加到文件中，而不会减慢训练速度。

这段代码主要是为了在训练过程中记录并可视化模型的性能指标，例如损失值、准确率等。
'''

# 终端运行
pip install tensorboard

tensorboard --logdir=./run --port 8088

# demo.py 
import torch
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('runs')
for i in SummaryWriter.__dict__.keys():
    if i.startswith("add_"):
        print(i)


writer = SummaryWriter('runs/add_scalar')
for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
writer.close()
```

教程链接：

- <https://blog.csdn.net/qq_44643484/article/details/120545860>

- <https://www.cnblogs.com/chenhuabin/p/16993006.html>
