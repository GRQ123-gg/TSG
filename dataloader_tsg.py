from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import lmdb #这行代码导入了lmdb模块，用于处理LMDB（Lightning Memory-Mapped Database）数据库，LMDB是一种高效的键值存储数据库。
import os
import numpy as np
import numpy.random as npr
import random

import torch
import torch.utils.data as data

import multiprocessing
import six  #six是一个用于处理Python 2和Python 3兼容性的库。

#这些导入语句使得代码可以使用这些模块和库中提供的函数和数据结构，以便进行数据处理、科学计算、深度学习和并行编程等任务。

class HybridLoader:
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.
    """
    def __init__(self, db_path, ext):  #用于加载数据
        self.db_path = db_path  #数据的路径
        self.ext = ext  #数据文件的扩展名
        if self.ext == '.npy':
            if self.db_path=='data/coco_pred_sg':
                self.loader = lambda x: np.load(x,allow_pickle=True,encoding="latin1").item()
            else:
                self.loader = lambda x: np.load(x,allow_pickle=True,encoding="latin1").item()
        else:
            self.loader = lambda x: np.load(x)['feat']

#这段代码根据数据文件的扩展名ext来确定加载数据的方法。
        # 如果扩展名是.npy，则使用np.load函数加载数据，并将其转换为字典类型。
        # 如果db_path是data/coco_pred_sg，则加载的数据是一个字典类型的对象。
        # 否则，加载的数据是一个字典类型的对象。如果扩展名不是.npy，则使用np.load函数加载数据，并获取其中的feat字段。
        if db_path.endswith('.lmdb'):
            self.db_type = 'lmdb'
            self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                                readonly=True, lock=False,
                                readahead=False, meminit=False)
        elif db_path.endswith('.pth'): # Assume a key,value dictionary
            self.db_type = 'pth'
            self.feat_file = torch.load(db_path)
            self.loader = lambda x: x
            print('HybridLoader: ext is ignored')
        elif db_path.endswith('h5'):
            self.db_type = 'h5'
            self.loader = lambda x: np.array(x).astype('float32')
        else:
            self.db_type = 'dir'
 #这段代码根据db_path的扩展名来确定数据的类型和加载方法。
# 如果扩展名是.lmdb，则表示数据存储在LMDB数据库中。
# 代码会使用lmdb.open函数打开数据库，并将其赋值给成员变量env。
# 如果扩展名是.pth，则表示数据是一个PyTorch的.pth文件，代码会使用torch.load函数加载数据，
# 并将其赋值给成员变量feat_file。加载方法也会被修改为一个无操作的lambda函数。
# 如果扩展名是.h5，则表示数据存储在HDF5文件中，代码会将加载方法修改为将数据转换为float32类型的lambda函数。
# 如果扩展名不满足上述条件，则表示数据存储在普通的文件夹中。

#这个HybridLoader类的作用是根据数据的路径和扩展名，选择合适的加载方法来读取数据。它支持.npy、LMDB、.pth和.h5等不同类型的数据文件，并提供了相应的加载逻辑。
    def get(self, key):

        if self.db_type == 'lmdb':
            env = self.env
            with env.begin(write=False) as txn:
                byteflow = txn.get(key.encode())
            f_input = six.BytesIO(byteflow)

    #如果数据类型是LMDB（即self.db_type为'lmdb'），则首先获取LMDB环境（env），
        # 然后在只读事务（txn）中使用给定的键（key）获取对应的字节流（byteflow）。
        # 接下来，将字节流转换为BytesIO对象（f_input），以便后续加载数据。
        elif self.db_type == 'pth':
            f_input = self.feat_file[key]
#如果数据类型是.pth（即self.db_type为'pth'），则直接通过键（key）从self.feat_file中获取对应的数据。
        elif self.db_type == 'h5':
            f_input = h5py.File(self.db_path, 'r')[key]
#如果数据类型是.h5（即self.db_type为'h5'），则通过h5py.File函数打开HDF5文件，并通过键（key）获取对应的数据。
        else:
            f_input = os.path.join(self.db_path, key + self.ext)
#如果数据类型是普通文件夹（即self.db_type为'dir'），则根据数据路径、键和扩展名构建文件路径。
        # load image
        feat = self.loader(f_input)
#最后，通过调用加载器（self.loader）来加载数据。加载器接受文件路径作为输入，并返回加载的数据。
        return feat
#根据数据路径和扩展名的不同，实现了不同的加载方式。可以根据数据库类型（LMDB、.pth、HDF5文件或普通文件夹）使用相应的方法来获取数据
class Dataset(data.Dataset):
    
    def get_vocab_size(self):  #表示词汇表的大小
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word  #序号到词语的映射（词汇表）

    def get_seq_length(self):
        return self.seq_length  #序列的长度

    def __init__(self, opt):
        self.opt = opt
        self.seq_per_img = opt.seq_per_img
        
        # feature related options
        self.use_fc = getattr(opt, 'use_fc', True)
        self.use_att = getattr(opt, 'use_att', True)
        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)
#这几行代码根据opt中的属性值来初始化类的成员变量。
        # getattr函数用于获取对象的属性值，如果属性不存在，则返回指定的默认值。这些成员变量用于控制数据集的特征相关选项。
        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        if 'ix_to_word' in self.info:
            self.ix_to_word = self.info['ix_to_word']
            self.vocab_size = len(self.ix_to_word)
            print('vocab size is ', self.vocab_size)
#这段代码加载包含数据集附加信息的JSON文件。
# 通过读取JSON文件，将其中的ix_to_word字段赋值给self.ix_to_word，并计算词汇表的大小（vocab_size）。这些信息将用于后续的数据处理。
        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_box_dir, opt.input_label_h5)
#这段代码打印加载HDF5文件的相关信息，包括特征文件、注意力文件、边界框文件和标签文件的路径。

#这个Dataset类用于处理数据集，并提供了获取词汇表大小、词汇表和序列长度的方法。它还在初始化过程中加载了附加信息的JSON文件，并打印了加载HDF5文件的相关信息。
        #对数据集进行配置和加载相关文件，包括JSON文件和HDF5文件。还会根据数据集的信息设置词汇表的大小和词汇表本身。
        """
        Setting input_label_h5 to none is used when only doing generation.
        For example, when you need to test on coco test set.
        """
        if self.opt.input_label_h5 != 'none':  #如果 self.opt.input_label_h5 不等于 'none'，则执行以下操作
            self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
    #使用 h5py.File 函数以只读模式打开指定的 HDF5 文件 (self.opt.input_label_h5)。使用 driver='core' 参数将数据完全加载到内存中。
            # load in the sequence data
            seq_size = self.h5_label_file['labels'].shape
    #获取序列数据的大小，并将其存储在 seq_size 中。
            self.label = self.h5_label_file['labels'][:]
    #将标签数据加载到 self.label 中。
            self.seq_length = seq_size[1]
    #将序列长度设置为 seq_size 的第二个维度的值。
            print('max sequence length in data is', self.seq_length)
    #打印出数据中的最大序列长度。
            # load the pointers in full to RAM (should be small enough)
            self.label_start_ix = self.h5_label_file['label_start_ix'][:]
            self.label_end_ix = self.h5_label_file['label_end_ix'][:]
    #将标签的起始索引和结束索引加载到 self.label_start_ix 和 self.label_end_ix 中。

        else:
            self.seq_length = 1
 #如果 self.opt.input_label_h5 等于 'none'，则将序列长度设置为 1。
        self.fc_loader = HybridLoader(self.opt.input_fc_dir, '.npy')
    #self.fc_loader 用于加载扩展名为 '.npy' 的图像特征数据，初始化时使用 self.opt.input_fc_dir 作为输入目录。
        self.att_loader = HybridLoader(self.opt.input_att_dir, '.npz')
    #self.att_loader 用于加载扩展名为 '.npz' 的注意力特征数据，初始化时使用 self.opt.input_att_dir 作为输入目录。
        self.rela_loader = HybridLoader(self.opt.input_rela_dir, '.npy')
    #self.rela_loader 用于加载扩展名为 '.npy' 的关系特征数据，初始化时使用 self.opt.input_rela_dir 作为输入目录。
        self.box_loader = HybridLoader(self.opt.input_box_dir, '.npy')
#self.box_loader 用于加载扩展名为 '.npy' 的边界框特征数据，初始化时使用 self.opt.input_box_dir 作为输入目录。
        self.num_images = len(self.info['images']) # self.label_start_ix.shape[0]
#获取图像的数量，通过 len(self.info['images']) 计算得到，并将其存储在 self.num_images 中。
        print('read %d image features' %(self.num_images))
#注意：self.info['images'] 是一个图像信息的列表。
 #这段代码根据self.opt.input_label_h5的值加载标签数据，并根据文件的扩展名使用HybridLoader对象加载不同类型的数据文件。此外，还获取了图像数量和序列的最大长度。
        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}

    #初始化一个名为split_ix的空字典，其中包含三个键：'train'、'val'和'test'。相应的值是空列表。
        for ix in range(len(self.info['images'])):
    #遍历self.info字典中的'images'列表的索引。
            img = self.info['images'][ix]
#从'images'列表中获取当前索引ix处的图像，并将其赋值给变量img。
            if not 'split' in img:
#检查img字典中是否不存在'split'键。这个条件适用于没有被分配拆分的图像。
                self.split_ix['train'].append(ix)
                self.split_ix['val'].append(ix)
                self.split_ix['test'].append(ix)
            elif img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: # restval
                self.split_ix['train'].append(ix)
#如果前面的条件都不成立，并且opt.train_only的值为0，则表示该图像属于'restval'拆分
#将当前索引ix添加到split_ix字典中的'train'拆分列表中。
        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))
#代码输出了三个打印语句，分别显示了分配给训练集、验证集和测试集的图像数量
    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
#获取图像索引ix对应的标题起始位置，并将其减1。label_start_ix从1开始计数，因此需要减去1以匹配Python的索引从0开始的规则。
        ix2 = self.label_end_ix[ix] - 1
#获取图像索引ix对应的标题结束位置，并将其减1。
        ncap = ix2 - ix1 + 1 # number of captions available for this image
#计算该图像可用的标题数量
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'
#断言语句，用于确保图像至少有一个标题。如果某个图像没有任何标题，会触发断言错误。
        if ncap < seq_per_img:
#如果可用的标题数量小于所需的序列数
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
#创建一个全零数组，用于存储子采样后的序列。数组的形状为(seq_per_img, self.seq_length)，数据类型为整型。
            for q in range(seq_per_img):
#循环seq_per_img次，进行子采样
                ixl = random.randint(ix1,ix2)
#在标题起始位置和结束位置之间随机选择一个索引值。
                seq[q, :] = self.label[ixl, :self.seq_length]
#将选择的标题序列赋值给seq数组的第q行。self.label[ixl, :self.seq_length]表示从self.label中获取第ixl行，并截取前self.seq_length列。
        else:
#如果可用的标题数量大于等于所需的序列数。
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
#在标题起始位置和结束位置之间随机选择一个索引值，保证可以获取到连续的seq_per_img个标题。
            seq = self.label[ixl: ixl + seq_per_img, :self.seq_length]
#从self.label中获取连续的seq_per_img行，并截取前self.seq_length列，作为子采样后的序列。

        return seq
#根据给定的图像索引ix，从数据集中获取对应图像的标注序列。如果要获取的标注序列数量大于图像实际拥有的标注序列数量，则进行随机采样，并返回获取的标注序列。
    def collate_func(self, batch, split): #用于将批量的样本数据进行整理和处理
        seq_per_img = self.seq_per_img #获取每张图像的序列数。

        fc_batch = []
        att_batch = []
        label_batch = []
        rela_batch = []
        rela_seq_len = []
        attr_len = []
        rela_len = []
        att_len = []
        rela_info = []
#fc_batch：用于存储图像特征的列表。
# att_batch：用于存储注意力图的列表。
# label_batch：用于存储标签序列的列表。
# rela_batch：用于存储关系序列的列表。
# rela_seq_len：用于存储关系序列的长度的列表。
# attr_len：用于存储属性矩阵长度的列表。
# rela_len：用于存储关系矩阵长度的列表。
# att_len：用于存储注意力图长度的列表。
# rela_info：用于存储关系信息的列表。
# wrapped：布尔变量，用于记录是否存在被截断的样本。
# infos：用于存储图像相关信息的列表。
# gts：用于存储真实标签的列表。
        wrapped = False

        infos = []
        gts = []

        for sample in batch:
            # fetch image
            # # 获取图像特征、注意力图、序列标签、关系信息等
            tmp_fc, tmp_att, tmp_seq, tmp_rela, \
                ix, it_pos_now, tmp_wrapped = sample
            if tmp_wrapped:
                wrapped = True

            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)
#遍历批次中的样本数据：
# tmp_fc, tmp_att, tmp_seq, tmp_rela, ix, it_pos_now, tmp_wrapped = sample：获取样本数据中的图像特征、注意力图、序列标签、关系信息等。
# if tmp_wrapped: wrapped = True：如果存在被截断的样本，则将wrapped设置为True。
# 将图像特征、注意力图等添加到对应的列表中：
# fc_batch.append(tmp_fc)：将图像特征添加到fc_batch列表中。
# att_batch.append(tmp_att)：将注意力图添加到att_batch列表中。
            #见TSG.txt 6.1)

            # # 获取关系矩阵和属性矩阵的长度
            rela_matrix = tmp_rela['rela_matrix']
            attr_matrix = tmp_rela['obj_attr']
            rela_seq_len.append(len(rela_matrix)+len(attr_matrix))
            attr_len.append(len(attr_matrix))
            rela_len.append(len(rela_matrix))
            att_len.append(len(tmp_att))
            # 获取关系矩阵和属性矩阵的长度：
            #
            # rela_matrix = tmp_rela['rela_matrix']：从关系信息中获取关系矩阵。
            # attr_matrix = tmp_rela['obj_attr']：从关系信息中获取属性矩阵。
            # rela_seq_len.append(len(rela_matrix) + len(attr_matrix))：将关系矩阵和属性矩阵的长度添加到rela_seq_len列表中。
            # attr_len.append(len(attr_matrix))：将属性矩阵的长度添加到attr_len列表中。
            # rela_len.append(len(rela_matrix))：将关系矩阵的长度添加到rela_len列表中。
            # att_len.append(len(tmp_att))：将注意力图的长度添加到att_len列表中。
            #将属性矩阵和关系矩阵转换为序列形式
            tmp_rela_seq = np.zeros([len(rela_matrix)+len(attr_matrix)*2], dtype = 'int')
            tmp_rela_seq[0:len(attr_matrix)]=attr_matrix[:,1]
            tmp_rela_seq[len(attr_matrix):2*len(attr_matrix)]=attr_matrix[:,2]
            tmp_rela_seq[2*len(attr_matrix):]=rela_matrix[:,2]
            rela_batch.append(tmp_rela_seq)
            rela_info.append(rela_matrix)
            # 将属性矩阵和关系矩阵转换为序列形式：
            #
            # 创建一个全零数组tmp_rela_seq，长度为关系矩阵长度加上属性矩阵长度的两倍。
            # tmp_rela_seq[0:len(attr_matrix)] = attr_matrix[:,
            #                                    1]：将属性矩阵的第二列（属性标签）复制到tmp_rela_seq的前半部分。
            # tmp_rela_seq[len(attr_matrix):2 * len(attr_matrix)] = attr_matrix[:,
            #                                                       2]：将属性矩阵的第三列（属性值）复制到tmp_rela_seq的后半部分。
            # tmp_rela_seq[2 * len(attr_matrix):] = rela_matrix[:,
            #                                       2]：将关系矩阵的第三列（关系标签）复制到tmp_rela_seq的最后部分。
            # rela_batch.append(tmp_rela_seq)：将转换后的关系序列添加到rela_batch列表中。
            # rela_info.append(rela_matrix)：将关系矩阵添加到rela_info列表中。
            #        # 创建标签序列
            tmp_label = np.zeros([seq_per_img, self.seq_length + 2], dtype = 'int')
            if hasattr(self, 'h5_label_file'):
                # if there is ground truth
                tmp_label[:, 1 : self.seq_length + 1] = tmp_seq
            label_batch.append(tmp_label)
#创建标签序列：
#
# 创建一个全零数组tmp_label，形状为(seq_per_img, self.seq_length + 2)，
# 其中seq_per_img为每张图像的序列数，self.seq_length + 2是序列的长度（加上开始和结束标记）。
# if hasattr(self, 'h5_label_file'):：检查是否存在真实标签。
# 如果有真实标签，将临时序列tmp_seq复制到tmp_label的对应位置。
# label_batch.append(tmp_label)：将标签序列添加到label_batch列表中。
            # Used for reward evaluation
            if hasattr(self, 'h5_label_file'):
                # if there is ground truth
                gts.append(self.label[self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
            else:
                gts.append([])
#用于奖励评估：
#
# if hasattr(self, 'h5_label_file'):：检查是否存在真实标签。
# 如果有真实标签，从整个标签集中切片出与当前样本对应的真实标签，并添加到gts列表中。
# 否则，将一个空列表添加到gts中。
            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix].get('file_path', '')
            infos.append(info_dict)
#记录相关信息：
#
# 创建一个字典info_dict，用于存储当前图像的相关信息。
# info_dict['ix'] = ix：将图像在数据集中的索引添加到info_dict中。
# info_dict['id'] = self.info['images'][ix]['id']：将图像的唯一标识符添加到info_dict中。
# info_dict['file_path'] = self.info['images'][ix].get('file_path', '')：
        # 将图像的文件路径添加到info_dict中，如果不存在文件路径，则设置为空字符串。
# infos.append(info_dict)：将info_dict添加到infos列表中。
        # #sort by att_feat length
        # fc_batch, att_batch, label_batch, gts, infos = \
        #     zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: len(x[1]), reverse=True))
        fc_batch, att_batch, rela_batch, label_batch, gts, infos = \
            zip(*sorted(zip(fc_batch, att_batch, rela_batch, label_batch, gts, infos), key=lambda x: 0, reverse=True))
        #第一行代码使用sorted函数和zip函数对fc_batch、att_batch、rela_batch、label_batch、gts和infos进行排序和重新组合，
        # 以确保它们按照fc_batch的值从大到小排列。
        data = {}
        #接下来，创建一个空字典data。
        data['fc_feats'] = np.stack(fc_batch)
        #data['fc_feats']键对应的值是将fc_batch列表中的元素堆叠成一个numpy数组。
        # merge att_feats
        max_att_len = 0
        for i in range(len(att_batch)):
            max_att_len = max(max_att_len, att_len[i]+rela_len[i]+attr_len[i])
#通过循环遍历att_batch列表，计算att_len[i] + rela_len[i] + attr_len[i]的最大值，并将结果存储在max_att_len中。
        data['att_feats'] = np.zeros([len(att_batch), max_att_len, att_batch[0].shape[1]], dtype = 'float32')
        for i in range(len(att_batch)):
            data['att_feats'][i, :att_batch[i].shape[0]] = att_batch[i]
#data['att_feats']键对应的值是一个形状为[len(att_batch), max_att_len, att_batch[0].shape[1]]的全零数组，
        # 其中的数据来自att_batch列表。通过循环遍历att_batch列表，将每个元素的数据复制到对应位置。
        max_rela_len = max([_.shape[0] for _ in rela_batch])
#计算rela_batch列表中每个元素的长度，并找到最大长度max_rela_len。
        data['rela_seq'] = np.zeros([len(rela_batch), max_rela_len], dtype='int64')
        for i in range(len(rela_batch)):
            data['rela_seq'][i,:len(rela_batch[i])] = rela_batch[i]

#data['rela_seq']键对应的值是一个形状为[len(rela_batch),
# max_rela_len]的全零数组，其中的数据来自rela_batch列表。通过循环遍历rela_batch列表，将每个元素的数据复制到对应位置。
#这段代码的主要目的是根据输入的数据生成一个字典data，其中包含了经过处理后的各个特征和标签数据。
        # data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        # for i in range(len(att_batch)):
        #     data['att_masks'][i, :att_batch[i].shape[0]] = 1
        # # set att_masks to None if attention features have same length
        # if data['att_masks'].sum() == data['att_masks'].size:
        #     data['att_masks'] = None

        max_rela_seq_len = max([_ for _ in rela_seq_len])
    #计算 rela_seq_len 列表中的最大值，将其赋值给变量 max_rela_seq_len。
        data['cross_masks'] = np.zeros(
            [data['att_feats'].shape[0], max_att_len], dtype='float32')
    #创建一个全零数组 data['cross_masks']，形状为 [data['att_feats'].shape[0], max_att_len]，数据类型为 float32。
        for i in range(len(att_batch)):
            data['cross_masks'][i, :att_batch[i].shape[0]+rela_seq_len[i]] = 1
#使用循环遍历 att_batch 列表的索引，对 data['cross_masks'] 进行赋值操作。
        # 对于每个索引 i，将 data['cross_masks'][i, :att_batch[i].shape[0]+rela_seq_len[i]] 的值设为 1。
        data['enc_self_masks'] = np.zeros(
            [data['att_feats'].shape[0], max_att_len, max_att_len], dtype='float32')
#创建一个全零数组 data['enc_self_masks']，形状为 [data['att_feats'].shape[0], max_att_len, max_att_len]，数据类型为 float32。
        for i in range(len(att_batch)):
            data['enc_self_masks'][i, :att_batch[i].shape[0],:att_batch[i].shape[0]] = 1
            for j in range(attr_len[i]):
                data['enc_self_masks'][i, j, att_batch[i].shape[0]+j] = 1
                data['enc_self_masks'][i, att_batch[i].shape[0]+j, j] = 1
            for j in range(rela_len[i]):
                data['enc_self_masks'][i, np.int32(rela_info[i][j][0]),att_batch[i].shape[0]+attr_len[i]+j] = 1
                data['enc_self_masks'][i, att_batch[i].shape[0]+attr_len[i]+j,np.int32(rela_info[i][j][0])] = 1
                data['enc_self_masks'][i, np.int32(rela_info[i][j][1]),att_batch[i].shape[0]+attr_len[i]+j] = 1
                data['enc_self_masks'][i, att_batch[i].shape[0]+attr_len[i]+j,np.int32(rela_info[i][j][1])] = 1
#使用循环遍历 att_batch 列表的索引，对 data['enc_self_masks'] 进行赋值操作。
# 对于每个索引 i，首先将 data['enc_self_masks'][i, :att_batch[i].shape[0],:att_batch[i].shape[0]] 的值设为 1，
        # 然后根据 attr_len 和 rela_len 的信息，将适当的位置的值设为 1。
        data['attr_len'] = attr_len
        data['rela_len'] = rela_len
        data['att_len'] = att_len

        data['labels'] = np.vstack(label_batch)
#将 attr_len 和 rela_len 分别赋值给 data['attr_len'] 和 data['rela_len']。
#
# 将 label_batch 列表中的元素按垂直方向堆叠，得到 data['labels']。
#
# 总之，这段代码根据一些长度信息和数据的形状，生成了一些掩码，并调整了数据的形状，以便后续处理和模型训练使用。
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, data['labels'])))
        #使用 lambda 函数和 map 函数计算 data['labels'] 中每个元素不为零的个数，并加上2。将结果转换成数组，并赋值给 nonzeros。
        mask_batch = np.zeros([data['labels'].shape[0], self.seq_length + 2], dtype = 'float32')
    #创建一个全零数组 mask_batch，形状为 [data['labels'].shape[0], self.seq_length + 2]，数据类型为 float32。
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        #使用循环遍历 mask_batch 的索引和对应的行，将每行的前 nonzeros[ix] 个元素设为 1。
        data['masks'] = mask_batch
    #将生成的 mask_batch 赋值给 data['masks']。
        data['labels'] = data['labels'].reshape(len(batch), seq_per_img, -1)
    #将 data['labels'] 调整形状为 [len(batch), seq_per_img, -1]，并重新赋值给 data['labels']
        data['masks'] = data['masks'].reshape(len(batch), seq_per_img, -1)
# 3将 data['masks'] 调整形状为 [len(batch), seq_per_img, -1]，并重新赋值给 data['masks']。
        data['gts'] = gts # all ground truth captions of each images
        #将 gts 赋值给 data['gts']，表示每个图像的所有真实标注。
        data['bounds'] = {'it_pos_now': it_pos_now, # the it_pos_now of the last sample
                          'it_max': len(self.split_ix[split]), 'wrapped': wrapped}

    #创建一个字典 data['bounds']，包含了一些边界信息，如当前样本的 it_pos_now，最大样本数 it_max，是否已经遍历完数据集 wrapped。
        data['infos'] = infos
# 3将 infos 赋值给 data['infos']，包含了一些额外的数据信息。
        data = {k:torch.from_numpy(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor
#使用字典推导式，将 data 中的所有值转换为 PyTorch 张量（如果原值为 NumPy 数组）。

# 返回处理后的 data。
        return data
#将输入的批量样本数据整理成模型训练所需的形式，并进行相应的数据转换和填充操作。具体的处理过程包括对特征数据的处理、生成标签的掩码、生成注意力掩码等
    def __getitem__(self, index):
    #这段代码是一个__getitem__方法的实现，用于处理索引index对应的数据。下面是对代码每一行的中文解释：
        """This function returns a tuple that is further passed to collate_fn
        """
        ix, it_pos_now, wrapped = index #self.split_ix[index]
    #该方法接受一个索引index作为输入，并返回一个元组，该元组将进一步传递给collate_fn函数。
        if self.use_att:
            att_feat = self.att_loader.get(str(self.info['images'][ix]['id']))
            # Reshape to K x C
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            if self.norm_att_feat:
                att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
            if self.use_box:
                box_feat = self.box_loader.get(str(self.info['images'][ix]['id']))
                # devided by image width and height
                x1,y1,x2,y2 = np.hsplit(box_feat, 4)
                h,w = self.info['images'][ix]['height'], self.info['images'][ix]['width']
                box_feat = np.hstack((x1/w, y1/h, x2/w, y2/h, (x2-x1)*(y2-y1)/(w*h))) # question? x2-x1+1??
                if self.norm_box_feat:
                    box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
                att_feat = np.hstack([att_feat, box_feat])
                # sort the features by the size of boxes
                att_feat = np.stack(sorted(att_feat, key=lambda x:x[-1], reverse=True))
        else:
            att_feat = np.zeros((0,0), dtype='float32')

#如果self.use_att为True，则加载与当前索引对应的注意力特征att_feat。注意力特征首先被重塑为形状为K x C的二维数组。
    # 如果self.norm_att_feat为True，则对注意力特征进行归一化处理。如果self.use_box为True，
    # 还会加载与当前索引对应的边界框特征box_feat。边界框特征的值会根据图像的宽度和高度进行归一化处理，
    # 并与注意力特征进行水平拼接。最后，根据边界框的大小对特征进行排序。如果self.use_att为False，则att_feat被设置为一个形状为(0, 0)的空数组。
        if self.use_fc:
            try:
                fc_feat = self.fc_loader.get(str(self.info['images'][ix]['id']))
            except:
                # Use average of attention when there is no fc provided (For bottomup feature)
                fc_feat = att_feat.mean(0)
        else:
            fc_feat = np.zeros((0), dtype='float32')
#如果self.use_fc为True，则加载与当前索引对应的全连接特征fc_feat。如果无法获取全连接特征，
    # 则使用注意力特征的平均值作为全连接特征（适用于底层特征）。如果self.use_fc为False，则fc_feat被设置为一个形状为(0)的空数组。
        if hasattr(self, 'h5_label_file'):
            seq = self.get_captions(ix, self.seq_per_img)
        else:
            seq = None
#如果存在属性h5_label_file，则调用get_captions方法获取与当前索引对应的标注序列seq，
    # 并指定每个图像的序列数量为self.seq_per_img。否则，将seq设置为None。
        rela_data = self.rela_loader.get(str(self.info['images'][ix]['id']))
#加载与当前索引对应的关系数据rela_data。
        return (fc_feat,
                att_feat, seq, rela_data,
                ix, it_pos_now, wrapped)
#返回一个元组，其中包含fc_feat（全连接特征）、att特（注意力特征）、
    # seq（标注序列）、rela_data（关系数据）、ix（索引）、it_pos_now（当前位置）、wrapped`（是否循环）等值。
#这段代码根据给定的索引从数据集中获取与图像相关的各种特征和数据。
    def __len__(self):
        return len(self.info['images'])
#返回图像数据集中图像的数量

class DataLoader:
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.dataset = Dataset(opt)
#DataLoader类的初始化方法，接受一个opt参数作为输入。opt是一个选项对象，包含了一些数据加载的配置信息。
# 在初始化过程中，将opt保存到self.opt中，并从opt中获取批处理大小batch_size。
        # 然后创建一个Dataset对象，将opt作为参数传递给Dataset的构造函数，得到一个数据集对象，并将其保存到self.dataset中。
        # Initialize loaders and iters
        self.loaders, self.iters = {}, {}
#初始化两个空字典loaders和iters，用于存储加载器和迭代器。
        for split in ['train', 'val', 'test']:
            if split == 'train':
                sampler = MySampler(self.dataset.split_ix[split], shuffle=True, wrap=True)
            else:
                sampler = MySampler(self.dataset.split_ix[split], shuffle=False, wrap=False)
#循环遍历三个数据集划分：'train'、'val'和'test'。根据当前划分，创建一个sampler对象。
# 如果当前划分是'train'，则创建一个可随机打乱并循环抽样的sampler；
# 否则创建一个不打乱且不循环的sampler。sampler的作用是定义数据加载的顺序，以及在遍历完一个epoch后是否循环回到数据集的开头。
            self.loaders[split] = data.DataLoader(dataset=self.dataset,
                                                  batch_size=self.batch_size,
                                                  sampler=sampler,
                                                  pin_memory=True,
                                                  # num_workers=4, # 4 is usually enough
                                                  collate_fn=lambda x: self.dataset.collate_func(x, split),
                                                  drop_last=False,
                                                  )
            self.iters[split] = iter(self.loaders[split])
#根据当前划分创建一个DataLoader对象，并将其保存到self.loaders字典中。
    # DataLoader用于加载数据，其中的参数包括数据集dataset、批处理大小batch_size、
    # 采样器sampler、是否将数据存入固定内存pin_memory、数据加载的工作进程数量num_workers（注释部分表示默认值为4），
    # 以及如何对每个批次的数据进行整理和处理的collate_fn函数。此外，还将DataLoader对象的迭代器保存到self.iters字典中，以便后续使用。

# 综上所述，该DataLoader类用于初始化数据加载器，根据数据集划分和配置参数创建相应的加载器和迭代器，便于以后进行数据的批量加载和访问。
#这段代码定义了一个DataLoader类，用于加载数据集并创建数据加载器。它根据传入的选项对象配置数据加载器的行为，
    # 包括批次大小、采样方式、线程数等。在初始化过程中，它创建了针对不同拆分的数据加载器和迭代器，以便在训练过程中逐批次地获取数据
    def get_batch(self, split):
        try:
            data = next(self.iters[split])
        except StopIteration:
            self.iters[split] = iter(self.loaders[split])
            data = next(self.iters[split])
        return data
#get_batch方法用于获取指定划分（split）的一个批次数据。首先尝试从self.iters[split]迭代器中获取下一个批次的数据。
    # 如果StopIteration异常被抛出，表示迭代器已经遍历完一个epoch，需要重置迭代器。然后重新获取迭代器的下一个批次数据。最后返回获取到的数据。
    def reset_iterator(self, split):
        self.loaders[split].sampler._reset_iter()
        self.iters[split] = iter(self.loaders[split])
#reset_iterator方法用于重置指定划分（split）的迭代器。通过调用sampler对象的_reset_iter方法来重置迭代器，并将迭代器重新赋值给self.iters[split]。
    def get_vocab_size(self):
        return self.dataset.get_vocab_size()
# 3get_vocab_size方法用于获取数据集的词汇表大小。它调用self.dataset的get_vocab_size方法来获取词汇表的大小，并将结果返回。
    @property
    def vocab_size(self):
        return self.get_vocab_size()
#vocab_size是一个装饰器，用于将get_vocab_size方法转化为一个只读属性。
    # 当访问self.vocab_size时，实际上是调用了get_vocab_size方法来获取词汇表的大小。
    def get_vocab(self):
        return self.dataset.get_vocab()
#get_vocab方法用于获取数据集的词汇表。它调用self.dataset的get_vocab方法来获取词汇表，并将结果返回。
    def get_seq_length(self):
        return self.dataset.get_seq_length()
#get_seq_length方法用于获取数据集中序列的长度。它调用self.dataset的get_seq_length方法来获取序列的长度，并将结果返回。
    @property
    def seq_length(self):
        return self.get_seq_length()
#seq_length方法用于获取序列的长度。它调用self.get_seq_length()方法来获取序列的长度，并将结果返回。
    def state_dict(self):
        def get_prefetch_num(split):
            if self.loaders[split].num_workers > 0:
                return (self.iters[split]._send_idx - self.iters[split]._rcvd_idx) * self.batch_size
            else:
                return 0
        return {split: loader.sampler.state_dict(get_prefetch_num(split)) \
                    for split, loader in self.loaders.items()}
#state_dict方法用于获取DataLoader的状态字典。首先定义了一个内部函数get_prefetch_num(split)，
    # 用于计算指定划分（split）的预取数量。如果加载器的工作进程数量大于0，则通过计算发送索引减去接收索引，
    # 并乘以批处理大小来估计预取的数据数量。否则返回0。然后使用字典推导式，遍历self.loaders中的每个划分和对应的加载器，
    # 调用加载器的sampler的state_dict方法，并将预取数量作为参数传递给state_dict方法。最终返回一个字典，其中键是划分，值是对应加载器sampler的状态字典。
    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        for split in self.loaders.keys():
            self.loaders[split].sampler.load_state_dict(state_dict[split])
#load_state_dict方法用于加载DataLoader的状态字典。首先检查传入的state_dict是否为None，
# 如果是则直接返回。然后遍历self.loaders中的每个划分，使用对应划分的加载器sampler调用load_state_dict方法，
# 传入state_dict[split]作为参数，以恢复加载器的状态。
#综上所述，这些方法提供了获取序列长度、保存和加载DataLoader的状态字典的功能，以便在需要时对DataLoader进行状态的保存和恢复。
#这两个方法一起提供了保存和加载DataLoader对象状态的功能，可以方便地保存和恢复数据加载器的状态，以便于在训练过程中断和重启时继续训练。
class MySampler(data.sampler.Sampler): #自定义的采样器
    def __init__(self, index_list, shuffle, wrap):
        self.index_list = index_list
        self.shuffle = shuffle
        self.wrap = wrap
        # if wrap, there will be not stop iteration called
        # wrap True used during training, and wrap False used during test.
        self._reset_iter()
#这是一个继承自 data.sampler.Sampler 的类 MySampler。
    # 在初始化方法 __init__ 中，它接受三个参数：index_list，shuffle 和 wrap。index_list 是索引列表，
    # 表示数据集中样本的索引顺序；shuffle 是一个布尔值，表示是否对索引进行随机洗牌；wrap 也是一个布尔值，
    # 用于控制采样器在遍历完所有样本后是否重新开始遍历。初始化方法中还调用了一个内部方法 _reset_iter()，用于初始化迭代器的计数器。
    def __iter__(self):
        return self

    def __next__(self):
        wrapped = False
        if self.iter_counter == len(self._index_list):
            self._reset_iter()
            if self.wrap:
                wrapped = True
            else:
                raise StopIteration()
        elem = (self._index_list[self.iter_counter], self.iter_counter+1, wrapped)
        self.iter_counter += 1
        return elem
#__next__ 方法用于获取下一个样本。首先检查迭代器的计数器是否等于索引列表的长度，如果相等，表示已经遍历完所有样本。
    # 在这种情况下，如果 wrap 为 True，即重新开始遍历，就调用 _reset_iter() 方法重置迭代器；
    # 如果 wrap 为 False，即不重新开始遍历，就抛出 StopIteration 异常来结束迭代。
    # 如果迭代器计数器不等于索引列表的长度，就获取当前计数器位置对应的索引，
    # 并构建一个元组 elem，包含了当前样本的索引、计数器值加一和一个表示是否重新开始遍历的布尔值。然后将迭代器计数器加一，并返回 elem。
    def next(self):
        return self.__next__()
#next 方法用于兼容 Python 2.x 的迭代器协议，它调用 __next__ 方法来获取下一个样本。

# 综上所述，这个自定义的采样器类 MySampler 提供了迭代遍历索引列表中样本的功能，
    # 支持是否随机洗牌和是否重新开始遍历的设置。在每次迭代中，它返回一个元组，包含当前样本的索引、计数器值和一个指示是否重新开始遍历的标志。
    def _reset_iter(self):
        if self.shuffle:
            rand_perm = npr.permutation(len(self.index_list))
            self._index_list = [self.index_list[_] for _ in rand_perm]
        else:
            self._index_list = self.index_list

        self.iter_counter = 0
#_reset_iter 方法用于重置迭代器状态。如果 shuffle 为 True，即需要对索引列表进行随机洗牌，
    # 则使用 npr.permutation 函数生成一个长度为索引列表长度的随机排列，
    # 并根据随机排列重新组织索引列表中的元素。如果 shuffle 为 False，则不进行洗牌，
    # 直接将索引列表赋值给 _index_list。然后将迭代器计数器 iter_counter 设置为0。
    def __len__(self):
        return len(self.index_list)
#__len__ 方法返回索引列表的长度，即样本的总数。
    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        self._index_list = state_dict['index_list']
        self.iter_counter = state_dict['iter_counter']
#load_state_dict 方法用于加载采样器的状态字典。
    # 如果传入的状态字典 state_dict 不为 None，
    # 则将状态字典中的 'index_list' 和 'iter_counter' 分别赋值给 _index_list 和 iter_counter。
    def state_dict(self, prefetched_num=None):
        prefetched_num = prefetched_num or 0
        return {
            'index_list': self._index_list,
            'iter_counter': self.iter_counter - prefetched_num
        }
#state_dict 方法用于获取采样器的状态字典。它接受一个可选的参数 prefetched_num，
# 默认值为0。返回一个包含 'index_list' 和 'iter_counter' 键的字典，
# 其中 'index_list' 对应 _index_list，'iter_counter' 对应 iter_counter 减去 prefetched_num。

# 综上所述，这些方法对自定义采样器类 MySampler 进行了补充和实现，包括重置迭代器状态、
# 获取样本总数、加载状态字典和获取状态字典的功能。这些方法用于管理和操作采样器的状态，以便在需要时保存和恢复采样器的状态。
    