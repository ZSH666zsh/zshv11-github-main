from ultralytics import YOLO
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/ZSH_Yaml/yolo11_PTeLU.yaml').load('yolo11s.pt')

    model.train(data=r'datasets\zshcoco128\zshcoco128.yaml',
                epochs=5,
                batch=16,

                project='runs/zsh',
                name='ZSH_DGBC0111',

                imgsz=640,
                device='0',
                task='detect',
                patience=50,
                )

"""
Y0LOv11 的超参数配置在 ultralytics/cfg文件夹下的 default.yaml 文件中

model参数:填入模型配置文件的路径，改进的话建议不填预训练模型权重
data参数:可以填入训练数据集配置文件的路径
imgsz参数:代表输入图像的尺寸，指定为 640x640 像素
epochs参数:代表训练的轮数
batch参数:代表批处理大小，电脑显存越大，就设置越大，根据自己电脑性能设置
workers参数:代表数据加载的工作线程数，出现显存爆了的话可以设置为0，默认是8
device参数:代表用哪个显卡训练，留空表示自动选择可用的GPU或CPU
optimizer参数:代表优化器类型
close mosaic参数:代表在多少个 epoch 后关闭 mosaic 数据增强
resume参数:代表是否从上一次中断的训练状态继续训练。设置为False表示从头开始新的训练。如果设置为True，则会加载上一次训练的模型权重和优化器状态，继续训练。这在训练被中断或在已有模型的基础上进行进一步训练时非常有用。
project参数:代表项目文件夹，用于保存训练结果
name参数:代表命名保存的结果文件夹
single_cls参数:代表是否将所有类别视为一个类别，设置为False表示保留原有类别
cache参数:代表是否缓存数据，设置为False表示不缓存。
"""

"""
关于模型结果随机化的问题：
1. 神经网络的权重在训练开始时是随机初始化的
2. 训练过程中使用了随机的数据增强技术（如随机裁剪、旋转、颜色抖动等）
3. 数据集加载时的随机打乱
4. 某些CUDA操作默认是非确定性的。PyTorch 底层的 cuDNN 库为了追求极致的计算速度，会使用一些非确定性 (non-deterministic) 的卷积算法。
5. 代码依赖于多个可能产生随机数的库，主要包括 PyTorch、NumPy 和 Python 自带的 random 库。

可在YAML文件添加以下参数确保结果可重现：
1. 设置固定的随机种子（seed=42）
2. 启用确定性模式（deterministic=True）
3. 将workers=0可以避免多进程数据加载带来的随机性，但会降低数据加载速度。

数字42本身没有特殊含义，只是一个常用的默认值。
在《银河系漫游指南》科幻小说中，42是"生命、宇宙以及一切"的终极答案，因此在计算机科学和编程领域中经常被用作默认的随机种子值。
在PyTorch和其他深度学习框架中，随机种子的取值范围通常是：32位有符号整数范围：从-2,147,483,648到2,147,483,647
实际中通常用非负整数，如0, 1, 42, 123, 1000等

seed作用：
1. 随机种子用于初始化随机数生成器，确保每次运行代码时生成相同的随机数序列。
2. 权重初始化：神经网络层的初始权重
3. 数据增强：随机裁剪、旋转、翻转等数据增强操作
4. 数据打乱：每个epoch中训练数据的顺序
5. Dropout：在训练过程中随机丢弃神经元
6. 优化器：某些优化器中的随机性（如随机采样）

可以只设置固定种子而不开启确定性模式，但这不能完全保证结果的一致性。
"""