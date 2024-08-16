class DefaultConfig(object):
    # env = 'default'       # Visdom环境
    model = 'SqueezeNet'    # 使用的模型，名字必须与models/__init__.py中的名字一致
    train_data_root = ''
    test_data_root = ''
    load_model_path = None

    batch_size = 1          # batch_size大小
    use_gpu = False
    num_workers = 4
    print_freq = 20         # 打印信息的间隔轮数

    debug_file = '/tmp/debug'
    result_file = 'result.csv'

    max_epoch = 10          # 训练轮数
    lr = 0.1                # 初始化学习率
    lr_decay = 0.95         # 学习率衰减，lr = lr * lr_decay
    weight_decay = 1e-4


