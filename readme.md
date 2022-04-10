# Numpy实现两层神经网络

## my_network.py

TwoLayerNetwork类实现了一个两层神经网络，隐藏层激活函数实现了ReLU和sigmoid。最后一层线性分类器后进行softmax，损失函数为cross entropy+regularization。

AdamOptimizer类实现了一个简易的Adam梯度下降优化器，可以实现学习率自适应下降。



training函数对指定的隐藏层大小hidden_num，l2正则系数reg，和学习率lr调用TwoLayerNetwork和AdamOptimizer类对模型训练。 MNIST数据集一共有70000个样本，实验中将54000样本用作训练集，6000样本作为验证集，10000样本作为测试集。对训练集和验证集采用梯度下降训练模型。该函数采用早停机制，在验证集准确度20个epoch不改善后停止训练。函数返回值为训练好的model和训练集验证集训练过程中的损失函数和准确率。



save_model和load_model采用pickle保存模型为二进制txt文件，也可以使用TwoLayerNetwork的save_parameter和load_parameter函数单纯保存模型numpy参数矩阵为npz文件。



## parameter_search.py

该脚本网格搜索了超参数空间隐藏层个数、l2正则系数、和学习率，搜索空间为

```Python
hiddens = [50,100,150,200,250]
regs = [0,0.001,0.01,0.1]
lrs = [0.0001,0.001,0.01]
```

直接运行脚本可以保存最好的模型，用于在测试集上测试



## visualize.ipynb

该notebook测试了验证集上最好的模型在测试集上的分类精度，并可视化训练和验证集的loss曲线，训练和验证集的accuracy曲线，以及可视化每层的参数。

