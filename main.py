import os.path
import models
import torch as t
from config import opt
from data.dataset import DogCat
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm


def train(**kwargs):
    """
    训练
    :param kwargs:
    :return:
    """
    # 根据命令行参数更新配置
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    # step1:定义网络模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # step2:数据预处理和加载
    train_data = DogCat(opt.train_data_root, mode="train")
    val_data = DogCat(opt.train_data_root, mode="val")
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size,
                                shuffle=False, num_workers=opt.num_workers)

    # step3:定义损失函数和优化器
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    weight_decay = opt.weight_decay
    optimizer = model.get_optimizer(lr=lr, weight_decay=weight_decay)

    # step4:计算指标，如平滑处理后之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10

    # 训练
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        # confusion_matrix.reset()

        for ii, (data, label) in enumerate(tqdm(train_dataloader)):

            # 训练模型参数
            input = data.to(opt.device)
            target = label.to(opt.device)

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # 更新统计指标以及可视化
            loss_meter.add(loss.item())
            # print("train_score.shape = ", score.shape)
            # print("train_score = ", score)
            # print("train_target.shape = ", target.shape)
            # print("train_target = ", target)
            confusion_matrix.add(score.detach(), target.detach())

            if ii % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])

                # 如果需要，则可以进入调试模式
                if os.path.exists(opt.debug_file):
                    import ipdb;
                    ipdb.set_trace()

        model.save()

        # # 计算验证集上的指标以及可视化
        val_cm, val_accuracy = val(model, val_dataloader)
        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch}, lr={lr}, loss:{loss}, train_cm:{train_cm}, val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], train_cm=str(confusion_matrix.value()), lr=lr,
            val_cm=str(val_cm.value())
        ))

        # 如果损失不再下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


@t.no_grad()
def val(model, dataloder):
    """
    计算模型在验证集上的准确率等信息，用于辅助训练
    :param model:
    :param dataloder:
    :return:
    """
    # 将模型设为验证模式
    model.eval()

    confusion_matrix = meter.ConfusionMeter(2)
    for ii, (val_input, label) in tqdm(enumerate(dataloder)):
        val_input = val_input.to(opt.device)
        score = model(val_input)

        # print("val_score.shape = ", score.shape)
        # print("val_score = ", score)
        # print("val_score.squeeze.shape = ", score.detach().squeeze().shape)
        # print("val_label.shape = ", label.long().shape)
        # print("val_label = ", label.long())

        confusion_matrix.add(score.detach(), label.detach().type(t.LongTensor))

    # 将模型恢复为训练模式
    model.train()

    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy


@t.no_grad()
def test(**kwargs):
    """
    测试（inference)
    :param kwargs:
    :return:
    """
    opt.parse(kwargs)

    # 模型加载
    model = getattr(models, opt.model)().eval
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # 数据加载
    test_data = DogCat(opt.test_data_root, mode="test")
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    results = []
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        input = data.to(opt.device)
        score = model(input)
        # 计算每个样本属于狗的概率
        probability = t.nn.functional.softmax(score)[:, 1].data.tolist()
        batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]
        results += batch_results
    write_csv(results, opt.result_file)
    return results


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)

def help():
    """
    打印帮助的信息： python file.py help
    """

    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire
    fire.Fire()
