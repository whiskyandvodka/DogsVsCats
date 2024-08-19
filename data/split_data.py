import os
import random
from shutil import copyfile
from tqdm import tqdm


def split_data(main_dir, training_dir, validation_dir, test_dir=None,
               include_test_split=True, split_size=0.8):
    """
    Splits the data into train validation and test sets(optional)

    :param main_dir: 包含图片的路径
    :param training_dir: 训练集路径
    :param validation_dir: 验证集路径
    :param test_dir: 测试集路径
    :param include_test_split: 是否包含测试集
    :param split_size: 拆分尺寸
    :return: 无返回值
    """

    files = []
    for file in os.listdir(main_dir):
        if os.path.getsize(os.path.join(main_dir, file)):
            files.append(file)

    shuffled_files = random.sample(files, len(files))
    split_train_val = int(split_size * len(files))
    train = shuffled_files[:split_train_val]
    split_val_test = int(split_train_val + (len(shuffled_files) - split_train_val)/2)

    if include_test_split:
        validation = shuffled_files[split_train_val:split_val_test]
        test = shuffled_files[split_val_test:]
    else:
        validation = shuffled_files[split_train_val:]

    class_name = main_dir.split('/')[-1]

    for ii, element in tqdm(enumerate(train)):
        copyfile(os.path.join(main_dir, element), os.path.join(training_dir, (class_name.lower() + '.' + element)))

    for ii, element in tqdm(enumerate(validation)):
        copyfile(os.path.join(main_dir, element), os.path.join(validation_dir, (class_name.lower() + '.' + element)))

    if include_test_split:
        for ii, element in tqdm(enumerate(test)):
            copyfile(os.path.join(main_dir, element), os.path.join(test_dir, element))

    print("Split successful!")


if __name__ == "__main__":
    split_data(
        r"d:/Deep_Learning\dateset\DogsVsCats\PetImages\Dog".replace('\\', '/'),
        r"d:\Deep_Learning\workspace\PyTorch_project\Dogs_Vs_Cats\cats-v-dogs\training".replace('\\', '/'),
        r"d:\Deep_Learning\workspace\PyTorch_project\Dogs_Vs_Cats\cats-v-dogs\val".replace('\\', '/'),
        r"d:\Deep_Learning\workspace\PyTorch_project\Dogs_Vs_Cats\cats-v-dogs\test".replace('\\', '/'),
        include_test_split=False, split_size=0.7
    )

    split_data(
        r"d:/Deep_Learning\dateset\DogsVsCats\PetImages\Cat".replace('\\', '/'),
        r"d:\Deep_Learning\workspace\PyTorch_project\Dogs_Vs_Cats\cats-v-dogs\training".replace('\\', '/'),
        r"d:\Deep_Learning\workspace\PyTorch_project\Dogs_Vs_Cats\cats-v-dogs\val".replace('\\', '/'),
        r"d:\Deep_Learning\workspace\PyTorch_project\Dogs_Vs_Cats\cats-v-dogs\test".replace('\\', '/'),
        include_test_split=False, split_size=0.7
    )

