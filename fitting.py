import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch.nn.functional as F
import scipy.io as sio

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        # 初始网络的内部结构
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # 一次正向行走过程
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

if __name__ == '__main__':

    # 训练集 - 生成并保存
    x = torch.unsqueeze(torch.linspace(-1,1,100), dim=1)
    y = x.pow(2) + 0.2*torch.rand(x.size())
    sio.savemat('data_x.mat',{"x":x.data.numpy()})
    sio.savemat('data_y.mat',{"y":y.data.numpy()})

    # 构建神经网络
    net = Net(n_feature=1, n_hidden=1000, n_output=1)
    print('网络结构为：', net)

    # 训练网络
    loss_func = F.mse_loss
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    for t in range(10000):
        # 使用全量数据 进行正向行走
        prediction = net(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()  # 清除上一梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 应用梯度

        # 间隔一段，对训练过程进行可视化展示
        if t % 1000 == 0:
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())  # 绘制真实曲线
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=' + str(loss.item()), fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

        # if t == 1000:
        #     plt.cla()
        #     plt.scatter(x.data.numpy(), y.data.numpy())  # 绘制真实曲线
        #     plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        #     plt.text(0.5, 0, 'Loss=' + str(loss.item()), fontdict={'size': 20, 'color': 'red'})
        #     sio.savemat('data_prediction.mat', {"prediction": prediction.data.numpy()})
    plt.ioff()
    plt.show()

    # 保存训练的模型
    torch.save(net,"fitting.pkl")

    # 打印最终的损失值
    net = torch.load("fitting.pkl")
    prediction = net(x)
    sio.savemat('data_prediction.mat',{"prediction":prediction.data.numpy()})
    loss = loss_func(prediction, y)
    print('Loss=' + str(loss.item()))

    # 可视化
    plt.cla()
    plt.plot(x.data.numpy(), y.data.numpy(),'ro', label="Origin data")
    plt.plot(x.data.numpy(), prediction.data.numpy(), label="Fitted data")
    plt.legend()
    plt.show()