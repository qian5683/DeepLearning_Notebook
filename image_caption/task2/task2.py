from keras.models import model_from_json
from PIL import Image as pil_image
from keras import backend as K
import numpy as np
from pickle import dump
from os import listdir
from keras.models import Model
import keras
import os


def load_vgg16_model():
    """从当前目录下面的 vgg16_exported.json 和 vgg16_exported.h5 两个文件中导入 VGG16 网络并返回创建的网络模型
    # Returns
        创建的网络模型 model
    """
    json_file = open("vgg16_exported.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()

    model=model_from_json(loaded_model_json)
    model.load_weights("vgg16_exported.h5")

    return model


def preprocess_input(x):
    """预处理图像用于网络输入, 将图像由RGB格式转为BGR格式.
       将图像的每一个图像通道减去其均值

    # Arguments
        x: numpy 数组, 4维.
        data_format: Data format of the image array.

    # Returns
        Preprocessed Numpy array.
    """
    # x=x.transpose((0,3,2,1))
    x=x[:,:,:,[2,1,0]]
    return (x-x.min())/(x.max()-x.min())


def load_img_as_np_array(path, target_size):
    """从给定文件加载图像,转换图像大小为给定target_size,返回32位浮点数numpy数组.

    # Arguments
        path: 图像文件路径
        target_size: 元组(图像高度, 图像宽度).

    # Returns
        A PIL Image instance.
    """
    img=pil_image.open(path)
    img=img.resize(target_size, pil_image.NEAREST)

    #K.floatx() 后端对应的数据类型，如tensorflow
    return np.asarray(img, dtype=K.floatx())


def extract_features(directory):
    """提取给定文件夹中所有图像的特征, 将提取的特征保存在文件features.pkl中,
       提取的特征保存在一个dict中, key为文件名(不带.jpg后缀), value为特征值[np.array]

    Args:
        directory: 包含jpg文件的文件夹

    Returns:
        features
    """
    model=load_vgg16_model()
    # pop the last layer
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    features = dict()
    for fn in listdir(directory):
        fn=directory+'/'+fn
        arr=load_img_as_np_array(fn, target_size=(224, 224))

        # 改变数组的形态，增加一个维度（批处理输入的维度）
        arr=arr.reshape((1,arr.shape[0],arr.shape[1],arr.shape[2]))

        # 预处理图像作为VGG模型的输入
        arr = preprocess_input(arr)

        # 计算特征
        feature =model.predict(arr, verbose=0)

        # 分离文件名和路径以及分离文件名和后缀
        (filepath, tempfilename) = os.path.split(fn)
        (filename, extension) = os.path.splitext(tempfilename)
        id=tempfilename
        print(id)
        features[id]=feature

    return features


if __name__ == '__main__':
    # 提取所有图像的特征，保存在一个文件中, 大约一小时的时间，最后的文件大小为127M
    directory = '../Flickr8k_Dataset/Flicker8k_Dataset'
    features = extract_features(directory)
    print('提取特征的文件个数：%d' % len(features))
    print(keras.backend.image_data_format())
    #保存特征到文件
    dump(features, open('features.pkl', 'wb'))



