from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from numpy import array
from pickle import load
from task5 import util
from pickle import dump
from collections import Counter #Counter 计数器
import re

def create_descriptions(filename):
    """
    去掉多余的序号 #0 1 2 3 4

    Args:
        filename: 文本文件

    Returns:
        filename: 文本文件
    """
    # with open(filename, 'r'), open("descriptions.txt", "w")as (file_read, f_write):
    with open(filename, 'r') as file_read:
        with open("descriptions.txt", "w")as f_write:
            for line in file_read:
                index = line.find('#')
                line_datas = line[:index]+' '+line[index+3:]
                f_write.write(line_datas)

def get_vocab_size(filename):
    doc = util.load_doc(filename)
    all_caption = list()
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the caption
        caption = line.split()[1:-1] # remove the '.' from the sentence
        all_caption=all_caption+caption
    vocab_size = len(Counter(all_caption))
    return vocab_size

def get_captain_max_length(filename):
    with open(filename,'r') as file_read:
        max = 0
        for line in file_read:
            line = line[:-1]  # 去除末尾的‘\n’
            # line = re.split('[ ,.]', line)
            line = line.split(" ")[1:]
            if max < len(line):
                max = len(line)
                print("the max is:", max, line)
    return max

def load_image_names(filename):
    image_names_list = list()
    with open(filename, 'r') as file_read:
        for line in file_read:
            image_names_list.append(line[:-1])  # line[:-1] 去掉行尾的‘\n’
    # print(image_names_list)
    return image_names_list

def creat_tokenizer():
    '''Convert text to number

    # https://keras.io/zh/preprocessing/text/#tokenizer

    # This is an Example:
    tokenizer = Tokenizer()
    lines = ['this is good', 'that is a cat']
    tokenizer.fit_on_texts(lines)
    results = tokenizer.texts_to_sequences(['cat is good'])
    print(results[0])
    '''

    tokenizer = Tokenizer()
    train_image_names = load_image_names('../Flickr8k_text/Flickr_8k.trainImages.txt')
    train_descriptions = util.load_clean_captions('descriptions.txt', train_image_names)
    lines = util.to_list(train_descriptions)
    print(lines)
    tokenizer.fit_on_texts(lines)
    dump(tokenizer, open('tokenizer.pkl', 'wb'))



def create_input_data(tokenizer, max_length, descriptions, photos_features, vocab_size):
    """
    从输入的图片标题list和图片特征构造LSTM的一组输入

    Args:
    :param tokenizer: 英文单词和整数转换的工具keras.preprocessing.text.Tokenizer
    :param max_length: 训练数据集中最长的标题的长度
    :param descriptions: dict, key 为图像的名(不带.jpg后缀), value 为list, 包含一个图像的几个不同的描述
    :param photos_features:  dict, key 为图像的名(不带.jpg后缀), value 为numpy array 图像的特征
    :param vocab_size: 训练集中表的单词数量
    :return: tuple:
            第一个元素为 numpy array, 元素为图像的特征, 它本身也是 numpy.array
            第二个元素为 numpy array, 元素为图像标题的前缀, 它自身也是 numpy.array
            第三个元素为 numpy array, 元素为图像标题的下一个单词(根据图像特征和标题的前缀产生) 也为numpy.array

    Examples:
        from pickle import loady
        tokenizer = load(open('tokenizer.pkl', 'rb'))
        max_length = 6
        descriptions = {'1235345':['startseq one bird on tree endseq', "startseq red bird on tree endseq"],
                        '1234546':['startseq one boy play water endseq', "startseq one boy run across water endseq"]}
        photo_features = {'1235345':[ 0.434,  0.534,  0.212,  0.98 ],
                          '1234546':[ 0.534,  0.634,  0.712,  0.28 ]}
        vocab_size = 7378
        print(create_input_data(tokenizer, max_length, descriptions, photo_features, vocab_size))
(array([[ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.434,  0.534,  0.212,  0.98 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ],
       [ 0.534,  0.634,  0.712,  0.28 ]]),
array([[  0,   0,   0,   0,   0,   2],
       [  0,   0,   0,   0,   2,  59],
       [  0,   0,   0,   2,  59, 254],
       [  0,   0,   2,  59, 254,   6],
       [  0,   2,  59, 254,   6, 134],
       [  0,   0,   0,   0,   0,   2],
       [  0,   0,   0,   0,   2,  26],
       [  0,   0,   0,   2,  26, 254],
       [  0,   0,   2,  26, 254,   6],
       [  0,   2,  26, 254,   6, 134],
       [  0,   0,   0,   0,   0,   2],
       [  0,   0,   0,   0,   2,  59],
       [  0,   0,   0,   2,  59,  16],
       [  0,   0,   2,  59,  16,  82],
       [  0,   2,  59,  16,  82,  24],
       [  0,   0,   0,   0,   0,   2],
       [  0,   0,   0,   0,   2,  59],
       [  0,   0,   0,   2,  59,  16],
       [  0,   0,   2,  59,  16, 165],
       [  0,   2,  59,  16, 165, 127],
       [  2,  59,  16, 165, 127,  24]]),
array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       ...,
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.]]))
    """
    X1, X2, y = list(), list(), list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                if photos_features.__contains__(key):
                    in_seq, out_seq = seq[:i], seq[i]
                    # 填充in_seq,使得其长度为max_length
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(photos_features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
    return array(X1), array(X2), array(y)

if __name__ == '__main__':
    # create_descriptions('../Flickr8k_text/Flickr8k.token.txt')

    # creat_tokenizer()

    tokenizer = load(open('tokenizer.pkl', 'rb'))
    # max_length = get_captain_max_length('descriptions.txt') # 训练数据集中最长的标题的长度
    # vocab_size = get_vocab_size('descriptions.txt') # 训练集中表的单词数量 len(tokenizer.word_index) + 1
    max_length = 38
    vocab_size = 9553
    train_image_names = load_image_names('../Flickr8k_text/Flickr_8k.trainImages.txt')
    train_descriptions = util.load_clean_captions('descriptions.txt', train_image_names)
    train_photo_features = util.load_photo_features('../task2/features.pkl', train_image_names)

    # print(create_input_data(tokenizer, max_length, train_descriptions, train_photo_features, vocab_size))
    # 全部内存放不下，取出一小部分放进去
    temp_des= dict()
    count = 0
    for k, v in train_descriptions.items():
        count += 1
        if count < 20:
            temp_des[k] = v
    print(create_input_data(tokenizer, max_length, temp_des, train_photo_features, vocab_size))
