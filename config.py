import warnings
import torch as t


class DefaultConfig(object):
    env = 'default'       # Visdom环境
    model = 'SqueezeNet'    # 使用的模型，名字必须与models/__init__.py中的名字一致
    train_data_root = r"d:\Deep_Learning\workspace\PyTorch_project\Dogs_Vs_Cats\cats-v-dogs\training".replace('\\', '/')
    test_data_root = r"d:\Deep_Learning\workspace\PyTorch_project\Dogs_Vs_Cats\cats-v-dogs\val".replace('\\', '/')
    load_model_path = None

    batch_size = 1          # batch_size大小
    use_gpu = False
    num_workers = 0
    print_freq = 20         # 打印信息的间隔轮数

    debug_file = '/tmp/debug'
    result_file = 'result.csv'

    max_epoch = 10          # 训练轮数
    lr = 0.001                # 初始化学习率
    lr_decay = 0.5         # 学习率衰减，lr = lr * lr_decay
    weight_decay = 0e-5
    device = 'cpu'

    def parse(self, kwargs):
        """
        根据字典kwargs更新配置参数
        :param kwargs:
        :return:
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
                setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        # 打印配置信息
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = DefaultConfig()



