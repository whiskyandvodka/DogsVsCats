import time

import torch as t

class BasicModule(t.nn.Module):
    """
    封装了nn.Module，主要提供load和save两个方法
    """
    def __init__(self):
        super().__init__()
        self.model_name = str(type(self))  # 模型的默认名字

    def load(self, path):
        """
        加载指定路径下的模型
        :param path: 模型所在路径
        :return:
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名+时间”作为文件名
        :param name: 模型名
        :return:
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '-'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        t.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay):
        return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class Flat(t.nn.Module):
    """
    把输入reshape成（batch_size,dim_length）
    """

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)


