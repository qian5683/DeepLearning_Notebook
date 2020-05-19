import util
import numpy as np
from pickle import load
from nltk.translate.bleu_score import corpus_bleu
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from task3 import task3

def word_for_id(integer, tokenizer):
    """
    将一个整数转换为英文单词
    :param integer: 一个代表英文的整数
    :param tokenizer: 一个预先产生的keras.preprocessing.text.Tokenizer
    :return: 输入整数对应的英文单词
    """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_caption(model, tokenizer, photo_feature, max_length = 38):
    """
    根据输入的图像特征产生图像的标题
    :param model: 预先训练好的图像标题生成神经网络模型
    :param tokenizer: 一个预先产生的keras.preprocessing.text.Tokenizer
    :param photo_feature:输入的图像特征, 为VGG16网络修改版产生的特征
    :param max_length: 训练数据中最长的图像标题的长度
    :return: 产生的图像的标题
    """
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])
        sequence = pad_sequences(sequence, maxlen=max_length)
        output = model.predict([photo_feature, sequence])
        integer = np.argmax(output)
        word = word_for_id(integer, tokenizer)
        # if the model is error
        if word is None:
            break
        in_text = in_text + " " + word
        if word == 'endseq':
            break

    return in_text

def generate_caption_run():
    # load test set
    filename = '../Flickr8k_text/Flickr_8k.testImages.txt'
    # filename = '../Flickr8k_text/Flickr_8k.trainImages.txt'
    # test = util.load_ids(filename)
    test = task3.load_image_names(filename)

    # photo feaatures
    test_features = util.load_photo_features('../task2/features.pkl', test)
    print('Photos: test=%d' % len(test_features))

    #load the model
    filename = '../task4/model_8.h5'
    model = load_model(filename)

    tokenizer = load(open('../task3/tokenizer.pkl', 'rb'))
    caption = generate_caption(model, tokenizer, test_features['2194286203_5dc620006a.jpg'], 38)

    return caption




def evaluate_model(model, captions, photo_features, tokenizer, max_length = 38):
    """计算训练好的神经网络产生的标题的质量,根据4个BLEU分数来评估

    Args:
        model:　训练好的产生标题的神经网络
        captions: dict, 测试数据集, key为文件名(不带.jpg后缀), value为图像标题list
        photo_features: dict, key为文件名(不带.jpg后缀), value为图像特征
        tokenizer: 英文单词和整数转换的工具keras.preprocessing.text.Tokenizer
        max_length：训练集中的标题的最大长度

    Returns:
        tuple:
            第一个元素为权重为(1.0, 0, 0, 0)的ＢＬＥＵ分数
            第二个元素为权重为(0.5, 0.5, 0, 0)的ＢＬＥＵ分数
            第三个元素为权重为(0.3333333, 0.3333333,0.3333333, 0)的ＢＬＥＵ分数
            第四个元素为权重为(0.25, 0.25, 0.25, 0.25)的ＢＬＥＵ分数

    """
    actual, predicted = list(), list()
    # step over the whole set
    for key, caption_list in captions.items():
        # generate description
        yhat = generate_caption(model, tokenizer, photo_features[key], max_length)
        # store actual and predicted
        references = [d.split() for d in caption_list]
        actual.append(references)
        predicted.append(yhat.split())

    # calcutate BLEU score
    blue1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    blue2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    blue3 = corpus_bleu(actual, predicted, weights=(0.3333333, 0.3333333,0.3333333, 0))
    blue4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    print("BLEU-1: %f" % blue1)
    print("BLEU-2: %f" % blue2)
    print("BLEU-3: %f" % blue3)
    print("BLEU-4: %f" % blue4)
    return blue1, blue2, blue3, blue4

def evaluate_model_run(model_name):
    # load test set
    filename = '../Flickr8k_text/Flickr_8k.testImages.txt'
    # test = util.load_ids(filename)
    test = task3.load_image_names(filename)
    test_captions = util.load_clean_captions('../task3/descriptions.txt', test)

    # photo feaatures
    test_features = util.load_photo_features('../task2/features.pkl', test)
    print('Photos: test=%d' % len(test_features))

    # load the model

    model = load_model(model_name)

    tokenizer = load(open('../task3/tokenizer.pkl', 'rb'))

    print(evaluate_model(model, test_captions, test_features, tokenizer))

if __name__ == '__main__':
    # use the model
    # print(generate_caption_run())
    for i in range(1, 20): 
        model_name='../task4/model_' + str(i) + '.h5'
        print(model_name)
        evaluate_model_run(model_name)






