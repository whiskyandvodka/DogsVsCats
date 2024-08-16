import models
from config import DefaultConfig


def train(**kwargs):
    """
    训练
    :param kwargs:
    :return:
    """
    # 根据命令行参数更新配置
    opt.parse(kwargs)

