import os
import random
from shutil import copy2

def split_dataset(file_path, file_path_label, new_file_path, new_file_path_label, split_rate):
    class_names = os.listdir(file_path)
    class_names_label = os.listdir(file_path_label)

    split_names = ['train', 'val', 'test']
    split_names_label = ['train', 'val', 'test']
    print(class_names)  # ['00000.jpg', '00001.jpg', '00002.jpg'... ]
    print(class_names_label)

    if not os.path.isdir(new_file_path):
        os.makedirs(new_file_path)
    if not os.path.isdir(new_file_path_label):
        os.makedirs(new_file_path_label)

    for split_name in split_names:
        split_path = os.path.join(new_file_path, split_name)
        split_path_label = os.path.join(new_file_path_label, split_name)
        print(split_path)
        print(split_path_label)
        if not os.path.isdir(split_path):
            os.makedirs(split_path)
        if not os.path.isdir(split_path_label):
            os.makedirs(split_path_label)

    # 按照比例划分数据集，并进行数据图片的复制
    for class_name in class_names:
        current_data_path = file_path
        current_data_path_label = file_path_label
        current_all_data = os.listdir(current_data_path)
        current_all_data_label = os.listdir(current_data_path_label)

        current_data_length = len(current_all_data)
        current_data_length_label = len(current_all_data_label)
        current_data_index_list = list(range(current_data_length))
        current_data_index_list_label = list(range(current_data_length_label))

        random.shuffle(current_data_index_list)
        random.shuffle(current_data_index_list_label)

        train_path = os.path.join(new_file_path, 'train/')
        train_path_label = os.path.join(new_file_path_label, 'train/')
        val_path = os.path.join(new_file_path, 'val/')
        val_path_label = os.path.join(new_file_path_label, 'val/')
        test_path = os.path.join(new_file_path, 'test/')
        test_path_label = os.path.join(new_file_path_label, 'test/')

        train_stop_flag = current_data_length * split_rate[0]
        train_stop_flag_label = current_data_length_label * split_rate[0]
        val_stop_flag = current_data_length * (split_rate[0] + split_rate[1])
        val_stop_flag_label = current_data_length_label * (split_rate[0] + split_rate[1])

    current_idx = 0
    train_num = 0
    val_num = 0
    test_num = 0

    for i in current_data_index_list:
        src_img_path = os.path.join(current_data_path, current_all_data[i])
        src_img_path_label = os.path.join(current_data_path_label, current_all_data_label[i])
        if current_idx < train_stop_flag:
            copy2(src_img_path, train_path)
            copy2(src_img_path_label, train_path_label)
            train_num += 1
        elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
            copy2(src_img_path, val_path)
            copy2(src_img_path_label, val_path_label)
            val_num += 1
        else:
            copy2(src_img_path, test_path)
            copy2(src_img_path_label, test_path_label)
            test_num += 1
        current_idx += 1

    print("Done!", train_num, val_num, test_num)


if __name__ == "__main__":
    # （需修改）原文件夹路径
    file_path = r"F:\data\ZSH_DataSets\zshtest\images"
    file_path_label = r"F:\data\ZSH_DataSets\zshtest\labels"
    # （需修改）新文件路径
    new_file_path = r"F:\data\ZSH_DataSets\zshtest\imgs"
    new_file_path_label = r"F:\data\ZSH_DataSets\zshtest\lbels"
    # （需修改）划分比例
    split_rate = [0.8, 0.1, 0.1]

    split_dataset(file_path, file_path_label, new_file_path, new_file_path_label, split_rate)
