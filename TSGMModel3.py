# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils

import copy
import math
import numpy as np

from .TSGMAttModel import AttModel

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, opt):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.opt = opt
        self.concat_type = getattr(opt, 'concat_type', 0)
        self.controller = getattr(opt, 'controller', 0)
        
    def forward(self, src, tgt, src_mask, src_tgt_mask, tgt_mask, rela_len, attr_len, att_len):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_tgt_mask,tgt, tgt_mask, rela_len, attr_len, att_len)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_tgt_mask, tgt, tgt_mask, rela_len, attr_len, att_len):
        memory_att, memory_attr, memory_rela, mask_att, mask_attr, mask_rela = divide_memory(memory, rela_len, attr_len, att_len)
        out = self.decoder(self.tgt_embed(tgt), memory_att, memory_attr, memory_rela, mask_att, mask_attr, mask_rela, tgt_mask)
        return out

def divide_memory(memory, rela_len, attr_len, att_len):
    seq_per_img = memory.shape[0] // len(rela_len)
    max_att_len = max([_ for _ in att_len])
    max_attr_len = max([_ for _ in attr_len])
    max_rela_len = max([_ for _ in rela_len])
    mask_att = torch.zeros([memory.shape[0],max_att_len], requires_grad=False).cuda()
    mask_attr = torch.zeros([memory.shape[0],max_attr_len], requires_grad=False).cuda()
    mask_rela = torch.zeros([memory.shape[0],max_rela_len], requires_grad=False).cuda()

    memory_att = torch.zeros([memory.shape[0], max_att_len, memory.shape[2]]).cuda()
    memory_attr = torch.zeros([memory.shape[0], max_attr_len, memory.shape[2]]).cuda()
    memory_rela = torch.zeros([memory.shape[0], max_rela_len, memory.shape[2]]).cuda()

    for i in range(len(rela_len)):
        mask_att[i*seq_per_img:(i+1)*seq_per_img,:att_len[i]] = 1
        mask_attr[i*seq_per_img:(i+1)*seq_per_img,:attr_len[i]] = 1
        mask_rela[i*seq_per_img:(i+1)*seq_per_img,:rela_len[i]] = 1

        memory_att[i*seq_per_img:(i+1)*seq_per_img,:att_len[i], :] = memory[i*seq_per_img:(i+1)*seq_per_img,:att_len[i],:]
        memory_attr[i*seq_per_img:(i+1)*seq_per_img,:attr_len[i], :] = memory[i*seq_per_img:(i+1)*seq_per_img,att_len[i]:attr_len[i]+att_len[i],:]
        memory_rela[i*seq_per_img:(i+1)*seq_per_img,:rela_len[i], :] = memory[i*seq_per_img:(i+1)*seq_per_img,attr_len[i]+att_len[i]:attr_len[i]+att_len[i]+rela_len[i],:]
    mask_att=mask_att.unsqueeze(1)
    mask_attr=mask_attr.unsqueeze(1)
    mask_rela=mask_rela.unsqueeze(1)
    return memory_att, memory_attr, memory_rela, mask_att, mask_attr, mask_rela

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory_att, memory_attr, memory_rela, mask_att, mask_attr, mask_rela, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory_att, memory_attr, memory_rela, mask_att, mask_attr, mask_rela, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn_att, src_attn_attr, src_attn_rela, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn_att = src_attn_att
        self.src_attn_attr = src_attn_attr
        self.src_attn_rela = src_attn_rela
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 5)
 
    def forward(self, x, memory_att, memory_attr, memory_rela, mask_att, mask_attr, mask_rela, tgt_mask):
        "Follow Figure 1 (right) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x_att = self.sublayer[1](x, lambda x: self.src_attn_att(x, memory_att, memory_att, mask_att))
        x_attr = self.sublayer[2](x, lambda x: self.src_attn_attr(x, memory_attr, memory_attr, mask_attr))
        x_rela = self.sublayer[3](x, lambda x: self.src_attn_rela(x, memory_rela, memory_rela, mask_rela))
        x = mod_controller(x_att, x_attr, x_rela, x)
        return self.sublayer[4](x, self.feed_forward)

def mod_controller(m_att, m_attr, m_rela, query):
    m = torch.stack((m_att, m_attr, m_rela), dim= 3) #m:50*17*512*3, query 50*17*512
    d_k = query.size(-1)
    query = query.unsqueeze(-1)
    scores = torch.matmul(query.transpose(-2, -1), m) / math.sqrt(d_k) #scores:50*17*3
    weights = F.softmax(scores, dim = -1) #scores:50*17*3
    out = torch.matmul(m, weights.transpose(-2, -1))
    out = out.squeeze(-1) #output:50*17*512*3
    return out

def subsequent_mask(size, all_former=0):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    if all_former:
        subsequent_mask = np.ones(attn_shape).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 1
    else:
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # print(scores.shape)
    # print(mask.shape)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TSGMModel3(AttModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1, opt={}):
        "Helper: Construct a model from hyperparameters."

        # 定义了一个名为make_model的方法，用于根据超参数构建模型。
        self.opt = opt #将opt赋值给类的opt属性。
        c = copy.deepcopy #将copy.deepcopy函数赋值给变量c，用于创建对象的深拷贝
        attn = MultiHeadedAttention(h, d_model, dropout)
        # 创建一个多头注意力机制的实例，参数为头数h、模型维度d_model和丢弃率dropout，并将其赋值给变量attn。
        attn_dec = MultiHeadedAttention(opt.h_dec, d_model, dropout)
        # 创建一个多头注意力机制的实例，参数为头数opt.h_dec、模型维度d_model和丢弃率dropout，并将其赋值给变量attn_dec。
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        # 创建一个位置前馈神经网络的实例，参数为模型维度d_model、前馈层维度d_ff和丢弃率dropout，并将其赋值给变量ff。
        position = PositionalEncoding(d_model, dropout)
        # 创建一个位置编码的实例，参数为模型维度d_model和丢弃率dropout，并将其赋值给变量position
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
            Decoder(DecoderLayer(d_model, c(attn), c(attn_dec), c(attn_dec), c(attn_dec),
                                 c(ff), dropout), N_dec),
            lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab), opt = self.opt)

        # 创建一个编码器-解码器模型的实例，参数包括编码器、解码器、源语言嵌入层、目标语言嵌入层、生成器以及其他可选参数。
        # Encoder(...)：创建一个编码器的实例，参数为编码器层、编码器的层数N_enc。
        #
        # EncoderLayer(...)：创建一个编码器层的实例，参数包括模型维度d_model、自注意力机制attn、前馈神经网络ff和丢弃率dropout。
        # Decoder(...)：创建一个解码器的实例，参数为解码器层、解码器的层数N_dec。
        #
        # DecoderLayer(...)：创建一个解码器层的实例，参数包括模型维度d_model、自注意力机制attn、
        # 源-目标注意力机制attn_dec、解码器-解码器注意力机制attn_dec、解码器-解码器注意力机制attn_dec、前馈神经网络ff和丢弃率dropout。
        # lambda x:x：一个恒等函数，用于传递输入数据。
        #
        # nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))：创建一个序列模型，包括目标语言的嵌入层和位置编码层。
        #
        # Generator(d_model, tgt_vocab)：创建一个生成器的实例，参数为模型维度d_model和目标语言词汇量tgt_vocab。
        #
        # opt = self.opt：将类的opt属性赋值给局部变量opt。
        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():  #对于模型中的每个参数，执行以下操作：
            if p.dim() > 1: #如果参数的维度大于1，表示是权重矩阵，则执行以下操作：
                nn.init.xavier_uniform_(p) #使用Glorot /在这段代码中，定义了一个名为TSGMModel3的类，它继承自AttModel类。
                # TSGMModel3类中有一个方法make_model，用于根据给定的超参数构建模型。
        return model
    # 在make_model方法中：
    #
    # self.opt = opt 将opt赋值给类的opt属性。
    #
    # c = copy.deepcopy 将copy.deepcopy赋值给变量c，用于创建对象的深拷贝。
    #
    # attn = MultiHeadedAttention(h, d_model, dropout) 创建一个名为attn的MultiHeadedAttention实例，其中h是头的数量，
    # d_model是模型的维度，dropout是丢弃率。
    #
    # attn_dec = MultiHeadedAttention(opt.h_dec, d_model, dropout) 创建一个名为attn_dec的MultiHeadedAttention实例，
    # 其中opt.h_dec是解码器头的数量。
    #
    # ff = PositionwiseFeedForward(d_model, d_ff, dropout) 创建一个名为ff的PositionwiseFeedForward实例，
    # 其中d_model是模型的维度，d_ff是前馈神经网络的维度。
    #
    # position = PositionalEncoding(d_model, dropout) 创建一个名为position的PositionalEncoding实例，其中d_model是模型的维度。
    #
    # model = EncoderDecoder(...) 创建一个名为model的EncoderDecoder实例，它由编码器、解码器、嵌入层和生成器组成。
    #
    # Encoder(...) 创建一个名为Encoder的实例，它由多个编码器层组成。
    #
    # EncoderLayer(...) 创建一个编码器层的实例，其中包含自注意力机制、前馈神经网络和丢弃率。
    # Decoder(...) 创建一个名为Decoder的实例，它由多个解码器层组成。
    #
    # DecoderLayer(...) 创建一个解码器层的实例，其中包含自注意力机制、源-目标注意力机制、解码器-解码器注意力机制、前馈神经网络和丢弃率。
    # lambda x:x 是一个恒等函数，用于传递输入数据。
    #
    # nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)) 创建一个序列模型，它包含目标语言的嵌入层和位置编码层。
    #
    # Generator(d_model, tgt_vocab) 创建一个名为Generator的实例，它用于生成模型的输出。
    #
    # opt = self.opt 将类的opt属性赋值给局部变量opt。
    #
    # for p in model.parameters(): 对模型的每个参数执行以下操作：
    #
    # if p.dim() > 1: 如果参数的维度大于1（即是权重矩阵），则执行以下操作：
    #
    # nn.init.xavier_uniform_(p) 使用Glorot / fan_avg初始化参数。
    # 最后，方法返回构建的模型。
    def __init__(self, opt):
        super(TSGMModel3, self).__init__(opt)
        self.opt = opt
        # 将opt赋值给类的opt属性。
        # self.config = yaml.load(open(opt.config_file))
        
        self.N_enc = getattr(opt, 'N_enc', opt.num_layers)
        # 从opt中获取N_enc属性的值，如果不存在则获取num_layers属性的值，并将结果赋值给类的N_enc属性。
        self.N_dec = getattr(opt, 'N_dec', opt.num_layers)
        # 从opt中获取N_dec属性的值，如果不存在则获取num_layers属性的值，并将结果赋值给类的N_dec属性。
        self.d_model = getattr(opt, 'd_model', opt.input_encoding_size)
        # 从opt中获取d_model属性的值，如果不存在则获取input_encoding_size属性的值，并将结果赋值给类的d_model属性。
        self.d_ff = getattr(opt, 'd_ff', opt.rnn_size)
        # 从opt中获取d_ff属性的值，如果不存在则获取rnn_size属性的值，并将结果赋值给类的d_ff属性。
        self.h = getattr(opt, 'num_att_heads', 8)
        # 从opt中获取num_att_heads属性的值，如果不存在则赋值为8，并将结果赋值给类的h属性
        self.dropout = getattr(opt, 'dropout', 0.1)
        # 从opt中获取dropout属性的值，如果不存在则赋值为0.1，并将结果赋值给类的dropout属性
        self.all_former = getattr(opt, 'all_former', 0)
        # 从opt中获取all_former属性的值，如果不存在则赋值为0，并将结果赋值给类的all_former属性。
        self.concat_type = getattr(opt, 'concat_type', 0)
        # 从opt中获取concat_type属性的值，如果不存在则赋值为0，并将结果赋值给类的concat_type属性。
        self.controller = getattr(opt, 'controller', 0)
        # 从opt中获取controller属性的值，如果不存在则赋值为0，并将结果赋值给类的controller属性

        delattr(self, 'att_embed')
        # 删除类的att_embed属性
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.d_model),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.d_model),) if self.use_bn==2 else ())))

        # 创建一个名为att_embed的nn.Sequential实例，其中包含一系列的线性层、激活函数和丢弃层。
        # 创建一个名为att_embed的nn.Sequential实例，其中包含一系列的线性层、激活函数和丢弃层。这些层用于将注意力特征嵌入到特定维度的向量空间中。
        self.rela_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Embedding(500, self.d_model),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.d_model),) if self.use_bn==2 else ())))
        # 创建一个名为rela_embed的nn.Sequential实例，其中包含一系列的线性层、激活函数和丢弃层。这些层用于将关系特征嵌入到特定维度的向量空间中。
        self.type_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Embedding(3, self.d_model),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.d_model),) if self.use_bn==2 else ())))
        #  创建一个名为type_embed的nn.Sequential实例，其中包含一系列的线性层、激活函数和丢弃层。这些层用于将类型特征嵌入到特定维度的向量空间中。
        
        delattr(self, 'embed')
        self.embed = lambda x : x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x
        delattr(self, 'logit')
        del self.ctx2att

        tgt_vocab = self.vocab_size + 1
        # 将self.vocab_size加1，并将结果赋值给变量tgt_vocab。

        self.model = self.make_model(0, tgt_vocab,
            N_enc=self.N_enc,
            N_dec=self.N_dec,
            d_model=self.d_model,
            d_ff=self.d_ff,
            h=self.h,
            dropout=self.dropout, opt = self.opt)
    # 调用make_model方法创建模型，并将结果赋值给类的model属性。
    # 调用make_model方法，传递一些参数，返回构建的模型。
    def logit(self, x): # unsafe way
        return self.model.generator.proj(x)
    # 定义了一个名为logit的方法，接受参数x，用于计算模型的输出。
    # 调用模型的generator.proj方法，传递参数x，返回模型的输出。
    def init_hidden(self, bsz):
        # 定义了一个名为init_hidden的方法，接受参数bsz，用于初始化隐藏状态。
        return []
    # 返回一个空列表作为隐藏状态的初始值。
    def _prepare_feature(self, att_feats, att_masks, enc_self_masks, rela_seq, rela_len, attr_len, att_len):
        # 定义了一个名为_prepare_feature的方法，用于准备特征数据。
        att_feats, seq, att_masks, seq_mask, enc_self_masks = self._prepare_feature_forward(att_feats, att_masks,
                                                                                            enc_self_masks, rela_seq,
                                                                                            rela_len, attr_len, att_len)
        memory = self.model.encode(att_feats, enc_self_masks)
        # 调用_prepare_feature_forward方法，传递一系列参数，
        # 返回特征数据的处理结果，并将结果分别赋值给att_feats、seq、att_masks、seq_mask和enc_self_masks。
        # 调用模型的encode方法，传递att_feats和enc_self_masks参数，返回编码后的特征向量，并将结果赋值给memory。
        return memory, att_masks, seq_mask, enc_self_masks
    # 返回memory、att_masks、seq_mask和enc_self_masks作为特征数据的准备结果。
    #
    # 最后，定义了几个辅助方法，其中logit用于计算模型的输出，init_hidden用于初始化隐藏状态，_prepare_feature用于准备特征数据。
    def _prepare_feature_forward(self, att_feats, att_masks, enc_self_masks, rela_seq, rela_len, attr_len, att_len, seq=None):
        # 定义了一个名为_prepare_feature_forward的方法，用于准备特征数据的进一步处理。
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        # 调用clip_att方法，传递att_feats和att_masks参数，对特征数据进行裁剪，并将裁剪后的结果分别赋值给att_feats和att_masks。
        # att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        att_feats = self.att_embed(att_feats) + self.type_embed(torch.ones(att_feats.shape[0:2],dtype=torch.int32).cuda()*0)
        # 将特征数据att_feats通过self.att_embed进行嵌入表示，然后加上一个类型嵌入表示，类型嵌入表示是通过self.type_embed对一个全为0的张量进行嵌入得到的。
        rela_feats = self.rela_embed(rela_seq)
        # 将关系序列数据rela_seq通过self.rela_embed进行嵌入表示，得到关系特征向量rela_feats。
        for i in range(len(rela_len)):
            # 遍历关系长度列表的索引。
            rela_type = self.type_embed(torch.ones(att_feats[i,att_len[i]:att_len[i]+attr_len[i],:].shape[0:1] ,dtype=torch.int32).cuda())
            # 根据关系特征的长度，创建一个全为1的张量，并通过self.type_embed进行嵌入表示，得到关系类型特征向量rela_type。
            att_feats[i,att_len[i]:att_len[i]+attr_len[i],:] = rela_feats[i,:attr_len[i],:] + rela_feats[i,attr_len[i]:2*attr_len[i],:] + rela_type
            # 将关系特征向量和关系类型特征向量加权相加，并将结果赋值给特征数据att_feats的相应位置。
            att_feats[i,att_len[i]+attr_len[i]:att_len[i]+attr_len[i]+rela_len[i], :] = rela_feats[i,2*attr_len[i]:2*attr_len[i]+rela_len[i],:] + \
                                                                                        self.type_embed(torch.ones(att_feats[i,att_len[i]+attr_len[i]:att_len[i]+attr_len[i]+rela_len[i], :].shape[0:1],dtype=torch.int32).cuda()*2)
        # 将关系特征向量的后半部分与类型嵌入表示加权相加，并将结果赋值给特征数据att_feats的相应位置。
        att_masks = att_masks.unsqueeze(-2)
        # 在att_masks的倒数第二个维度上增加一个维度。
        if seq is not None:
            # 判断seq是否为None。
            # crop the last one
            seq = seq[:,:-1]
            # 裁剪序列数据seq的最后一个元素。
            seq_mask = (seq.data > 0)
            # ：创建一个与seq形状相同的张量，其中大于0的元素为True，否则为False。
            seq_mask[:,0] = 1 # bos
            # 将序列掩码的第一个元素设置为1，表示序列的开始。
            seq_mask = seq_mask.unsqueeze(-2)
            # 在序列掩码的倒数第二个维度上增加一个维度。
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
            # 通过subsequent_mask方法生成一个下三角矩阵的掩码，并与序列掩码进行逐元素的逻辑与操作。
            seq_per_img = seq.shape[0] // att_feats.shape[0]
            # 计算每个图像对应的序列数。
            if seq_per_img > 1:
                # 如果每个图像对应的序列数大于1。
                att_feats, att_masks = utils.repeat_tensors(seq_per_img,
                                                            [att_feats, att_masks]
                                                            )
                # 通过utils.repeat_tensors方法将特征数据和掩码数据复制多份，使每个图像对应的特征数据和掩码数据都重复了seq_per_img次。
                enc_self_masks = utils.repeat_tensors(seq_per_img, enc_self_masks)
        #     通过utils.repeat_tensors方法将编码器自注意力掩码数据复制多份，使每个图像对应的编码器自注意力掩码数据都重复了seq_per_img次。
        else:
            # 如果每个图像对应的序列数不大于1
            seq_mask = None
        # 将序列掩码设置为None。
        #seq: 17, [0,7961,xxx,] seq_mask: 17, [[1,0,0],[1,1,0]]
        return att_feats, seq, att_masks, seq_mask, enc_self_masks
    # 返回处理后的特征数据att_feats、序列数据seq、特征掩码数据att_masks、序列掩码数据seq_mask和编码器自注意力掩码数据enc_self_masks作为结果。
    def _forward(self, fc_feats, att_feats, seq, att_masks, enc_self_masks, rela_seq, rela_len, attr_len, att_len):
        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
        att_feats, seq, att_masks, seq_mask, enc_self_masks = self._prepare_feature_forward(att_feats, att_masks, enc_self_masks, rela_seq, rela_len, attr_len, att_len, seq)

        out = self.model(att_feats, seq, enc_self_masks, att_masks, seq_mask, rela_len, attr_len, att_len)
        outputs = self.model.generator(out)
        return outputs
    # if seq.ndim == 3:  # B * seq_per_img * seq_len: 检查序列seq的维度是否为3，
    # 如果是，则将其形状重新调整为(-1, seq.shape[2])，其中-1表示根据其他维度的大小自动推断。
    #
    # att_feats, seq, att_masks, seq_mask, enc_self_masks =
    # self._prepare_feature_forward(att_feats, att_masks, enc_self_masks, rela_seq, rela_len, attr_len, att_len, seq):
    # 调用_prepare_feature_forward方法，传递特征数据att_feats、序列数据seq、特征掩码数据att_masks、编码器自注意力掩码数据enc_self_masks、关系序列数据rela_seq、关系长度数据rela_len、属性长度数据attr_len和特征长度数据att_len，对特征数据和掩码数据进行进一步的准备和处理。
    #
    # out = self.model(att_feats, seq, enc_self_masks, att_masks, seq_mask, rela_len, attr_len, att_len):
    # 调用self.model的__call__方法，传递特征数据att_feats、序列数据seq、编码器自注意力掩码数据enc_self_masks、特征掩码数据att_masks、
    # 序列掩码数据seq_mask、关系长度数据rela_len、属性长度数据attr_len和特征长度数据att_len，执行模型的前向计算，得到输出out。
    #
    # outputs = self.model.generator(out): 将输出out传递给模型的生成器self.model.generator，生成最终的预测输出outputs。
    #
    # return outputs: 返回最终的预测输出。
    def core(self, it, memory, state, mask, rela_len, attr_len, att_len):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask,
                               ys, 
                               subsequent_mask(ys.size(1), self.all_former)
                                        .to(memory.device), rela_len, attr_len, att_len)

        return out[:, -1], [ys.unsqueeze(0)]
    # if len(state) == 0: ys = it.unsqueeze(1): 如果state列表的长度为0，表示当前状态为空，将输入it在维度1上添加一个维度，得到ys。
    #
    # else: ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1):
    # 如果state列表的长度不为0，将当前输入it在维度1上添加一个维度，然后与state[0][0]在维度1上进行拼接，得到ys。
    #
    # out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1),
    # self.all_former).to(memory.device), rela_len, attr_len, att_len):
    # 调用self.model的decode方法，传递记忆memory、掩码mask、输入ys、
    # 后续掩码subsequent_mask(ys.size(1), self.all_former).to(memory.device)、关系长度数据rela_len、
    # 属性长度数据attr_len和特征长度数据att_len，执行解码器的前向计算，得到输出out。
    #
    # return out[:, -1], [ys.unsqueeze(0)]: 返回out的最后一个时间步的输出（即预测的下一个序列元素）
    # 以及更新后的状态[ys.unsqueeze(0)]，其中状态由输入ys在维度0上添加一个维度构成。
    def _sample(self, fc_feats, att_feats, att_masks, enc_self_masks, rela_seq, rela_len, attr_len, att_len, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        sample_n = int(opt.get('sample_n', 1))
        group_size = opt.get('group_size', 1)
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
#从opt字典中获取采样方法sample_method，默认为'greedy'。
# 从opt字典中获取束搜索的束大小beam_size，默认为1。
# 从opt字典中获取温度参数temperature，默认为1.0。
# 从opt字典中获取采样数量sample_n，默认为1。
# 从opt字典中获取分组大小group_size，默认为1。
# 从opt字典中获取是否输出log softmax概率output_logsoftmax，默认为1。
# 从opt字典中获取解码约束decoding_constraint，默认为0。
# 从opt字典中获取是否阻止三元组block_trigrams，默认为0。
# 从opt字典中获取是否移除不良结局remove_bad_endings，默认为0。
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, enc_self_masks, rela_seq, rela_len, attr_len, att_len,  opt)
        if group_size > 1:
            return self._diverse_sample(fc_feats, att_feats, att_masks, opt)
        #
        # 如果束搜索的束大小beam_size大于1，则调用self._sample_beam方法进行束搜索采样，并返回结果。
        # 如果分组大小group_size大于1，则调用self._diverse_sample方法进行多样性采样，并返回结果。
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size * sample_n)
        att_feats, att_masks, seq_mask, enc_self_masks = self._prepare_feature(att_feats, att_masks,enc_self_masks, rela_seq,rela_len, attr_len, att_len)

        if sample_n > 1:
           att_feats, att_masks = utils.repeat_tensors(sample_n,
                                                       [att_feats, att_masks]
                                                       )

        trigrams = []  # will be a list of batch_size dictionaries

        seq = fc_feats.new_zeros((batch_size * sample_n, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.seq_length, self.vocab_size + 1)
        # 获取批次大小batch_size，并利用sample_n进行扩展，得到新的扩展后的批次大小。
        # 初始化隐状态state，并将其形状调整为(batch_size * sample_n)。
        # 调用self._prepare_feature方法，准备特征数据和掩码数据，并获取处理后的特征数据att_feats、
        # 特征掩码数据att_masks、序列掩码数据seq_mask和编码器自注意力掩码数据enc_self_masks。
        # 如果sample_n大于1，则使用utils.repeat_tensors方法对特征数据att_feats和特征掩码数据att_masks进行复制，以扩展样本数量。
        # 初始化一个空列表trigrams，用于存储每个批次的三元组信息。
        # 创建一个形状为(batch_size * sample_n, self.seq_length)的全零张量seq，用于存储生成的序列。
        # 创建一个形状为(batch_size * sample_n, self.seq_length, self.vocab_size + 1)的全零张量seqLogprobs，用于存储生成序列的log概率。
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size * sample_n, dtype=torch.long)
            # 对于每个时间步t，如果t为0，表示输入的是序列的起始符号（<bos>），创建一个全零张量it，形状为(batch_size * sample_n)，数据类型为torch.long。
            logprobs, state = self.get_logprobs_state(it, att_feats, att_masks, state, rela_len, attr_len, att_len,
                                                      output_logsoftmax=output_logsoftmax)
            # 调用self.get_logprobs_state方法，传入输入it、特征数据att_feats、
            # 特征掩码数据att_masks、隐状态state以及其他相关参数，返回生成的log概率logprobs和更新后的隐状态state。
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp
            # 如果设置了解码约束decoding_constraint并且当前时间步t大于0，则进行解码约束处理。
            # 创建一个与logprobs形状相同的全零张量tmp。
            # 使用seq[:, t - 1]选择前一个时间步的生成结果，将其转化为张量，并在第1维上进行散射操作，将对应位置的概率值设为负无穷（-inf）。
            # 将tmp与logprobs相加，以实现解码约束。
            if remove_bad_endings and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                prev_bad = np.isin(seq[:, t - 1].data.cpu().numpy(), self.bad_endings_ix)
                # Make it impossible to generate bad_endings
                tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                logprobs = logprobs + tmp
            # 如果设置了移除不良结局remove_bad_endings并且当前时间步t大于0，则进行移除不良结局的处理。
            # 创建一个与logprobs形状相同的全零张量tmp。
            # 使用seq[:, t - 1]选择前一个时间步的生成结果，并将其转化为NumPy数组。
            # 使用np.isin函数判断前一个时间步的生成结果是否属于不良结局的索引集合self.bad_endings_ix，得到一个布尔数组prev_bad。
            # 将prev_bad转化为torch.Tensor类型，并在第0维上选择为True的位置，将对应位置的概率值设为负无穷（-inf）。
            # 将tmp与logprobs相加，以实现移除不良结局的效果。
            # Mess with trigrams
            # Copy from https://github.com/lukemelas/image-paragraph-captioning
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:, t - 3:t - 1]
                for i in range(batch_size):  # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current = seq[i][t - 1]
                    if t == 3:  # initialize
                        trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]:  # add to list
                            trigrams[i][prev_two].append(current)
                        else:  # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:, t - 2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda()  # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i, j] += 1
                # Apply mask to log probs
                # logprobs = logprobs - (mask * 1e9)
                alpha = 2.0  # = 4
                logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)
            # 如果设置了阻止三元组block_trigrams并且当前时间步t大于等于3，则进行阻止三元组的处理。
            # 从序列seq中选择前三个时间步（t - 3到t - 1）的生成结果，保存在prev_two_batch中。
            # 对于每个样本，获取前两个生成结果prev_two和当前时间步的生成结果current。
            # 如果t为3，表示初始化阶段，将prev_two作为键，将current作为值构建一个字典，并添加到trigrams列表中。
            # 如果t大于3，表示已经生成了至少两个时间步的序列，如果prev_two已经存在于trigrams[i]中，
            # 则将current添加到对应的列表中；如果prev_two不存在于trigrams[i]中，则创建一个新的列表，并将current添加到其中。
            # 从序列seq中选择前两个时间步（t - 2到t）的生成结果，保存在prev_two_batch中。
            # 创建一个形状与logprobs相同的全零张量mask，用于存储阻止的三元组信息，形状为batch_size x vocab_size。
            # 对于每个样本，获取前两个生成结果prev_two。
            # 如果prev_two存在于trigrams[i]中，则遍历该三元组列表中的每个元素j，将mask[i, j]的值加1。
            # 将mask应用于logprobs，以阻止出现已经生成过的三元组。在代码注释中，还给出了另外一种阻止方法的示例，即将logprobs减去一个非常大的值（mask * 1e9）。
            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)
            # 对于下一个单词的采样。
            # 如果当前时间步t等于最大序列长度self.seq_length，则跳出循环，不再进行生成。
            # 调用self.sample_next_word方法，传入logprobs、采样方法sample_method和温度参数temperature，
            # 返回采样的单词it和对应的log概率sampleLogprobs。
            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            # 如果当前时间步`t`为0，表示刚开始生成序列，将`unfinished`初始化为`it > 0`，
            # 即将未完成的序列标记为`True`，已完成的序列标记为`False`。
            # - 如果当前时间步`t`不为0，即已经生成了至少一个时间步的序列，则将`unfinished`与`(it > 0)`逐元素相乘，
            # 更新`unfinished`，将已完成的序列继续标记为`False`，未完成的序列保持不变。
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            seqLogprobs[:, t] = logprobs
            # quit loop if all sequences have finished
            # 将完成的序列（it * unfinished.type_as(it)）与未完成的序列进行逐元素相乘，将已完成的序列的值保留，未完成的序列被置为0。
            # 将生成的单词it赋值给序列seq的第t个时间步。
            # 将logprobs赋值给序列seqLogprobs的第t个时间步。
            if unfinished.sum() == 0:
                break
        # 如果所有序列都已经完成（即unfinished中的元素和为0），则跳出循环，停止生成
        return seq, seqLogprobs
    # 返回生成的序列seq和对应的log概率seqLogprobs。
    def _sample_beam(self, fc_feats, att_feats, att_masks, enc_self_masks, rela_seq, rela_len, attr_len, att_len, opt={}):
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        sample_n = opt.get('sample_n', 10)
        # 这些行从opt字典中获取beam_size、group_size和sample_n的值。如果这些键在字典中不存在，则使用默认值10、1和10。
        # when sample_n == beam_size then each beam is a sample.
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        # 这一行检查sample_n的值是否满足beam搜索的条件。sample_n要么等于1，要么等于beam_size除以group_size。如果条件不满足，将引发一个断言错误。
        batch_size = fc_feats.size(0)
        # 这一行从fc_feats张量的大小中获取批次大小
        att_feats, att_masks, seq_mask, enc_self_masks = self._prepare_feature(att_feats, att_masks, enc_self_masks,
                                                                               rela_seq, rela_len, attr_len, att_len)
        # 这一行调用_prepare_feature方法对输入特征和掩码进行预处理。
        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        # 这一行检查beam大小是否小于等于词汇表大小加一。如果条件不满足，将引发一个断言错误。
        seq = fc_feats.new_zeros((batch_size * sample_n, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.seq_length, self.vocab_size + 1)
        # lets process every image independently for now, for simplicity
        # 这些行用零初始化张量seq和seqLogprobs。seq的形状是(batch_size * sample_n, self.seq_length)，
        # 数据类型是torch.long。seqLogprobs的形状是(batch_size * sample_n, self.seq_length, self.vocab_size + 1)。
        self.done_beams = [[] for _ in range(batch_size)]
        # 这一行为每个批次项初始化空列表的done_beams属性。
        state = self.init_hidden(batch_size)
        # 这一行使用类的init_hidden方法初始化隐藏状态state。
        # first step, feed bos
        it = fc_feats.new_zeros([batch_size], dtype=torch.long)
        # 这一行用零初始化it张量。it的形状是[batch_size]，数据类型是torch.long。

        logprobs, state = self.get_logprobs_state(it, att_feats, att_masks, state, rela_len, attr_len, att_len)
        # 这一行调用get_logprobs_state方法，根据当前输入it和隐藏状态state计算下一个词的对数概率。它还更新了隐藏状态state。
        att_feats, att_masks = utils.repeat_tensors(beam_size,
                                                    [att_feats, att_masks]
                                                    )
        # 这一行使用utils模块中的repeat_tensors函数，将att_feats和att_masks张量沿批次维度重复beam_size次。
        # logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
        # p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(beam_size,
        #                                                                           [p_fc_feats, p_att_feats,
        #                                                                            pp_att_feats, p_att_masks]
        #                                                                           )
        self.done_beams = self.beam_search(state, logprobs, rela_len, attr_len, att_len, att_feats, att_masks, opt=opt)
        # 这一行使用utils模块中的repeat_tensors函数，将att_feats和att_masks张量沿批次维度重复beam_size次。
        for k in range(batch_size):
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq_len = self.done_beams[k][_n]['seq'].shape[0]
                    seq[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['logps']
            else:
                seq_len = self.done_beams[k][0]['seq'].shape[0]
                seq[k, :seq_len] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
                seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq, seqLogprobs

    # 这一段代码遍历每个批次项并根据`sample_n`是否等于`beam_size`来赋值`seq`和`seqLogprobs`张量。如果`sample_n`等于`beam_size`，
    # 则使用`_n`的迭代器来遍历`sample_n`次，并将每个beam的序列和对应的对数概率赋值给`seq`和`seqLogprobs`。
    # 如果`sample_n`不等于`beam_size`，则只使用第一个beam的序列和对数概率赋值给`seq`和`seqLogprobs`。
    # 这一行返回seq和seqLogprobs张量，其中包含生成的序列及其对应的对数概率。

    def get_logprobs_state(self, it, att_feats, att_masks, state, rela_len, attr_len, att_len, output_logsoftmax=1):
        # 'it' contains a word index
        # 这段代码定义了一个名为get_logprobs_state的方法，接受it、att_feats、att_masks、state、rela_len、attr_len和att_len作为输入参数。
        #
        # 以下是代码的逐行解释：
        xt = self.embed(it)
        # 这一行使用嵌入层embed将输入it进行嵌入。xt是嵌入后的结果。
        output, state = self.core(xt, att_feats, state, att_masks, rela_len, attr_len, att_len)
        # 这一行将嵌入后的结果xt、输入特征att_feats、隐藏状态state、
        # 注意力掩码att_masks以及其他相关长度参数传递给核心模型core进行计算。output是计算得到的输出结果，state是更新后的隐藏状态。
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)
        # 这一段根据output_logsoftmax的值决定是否对输出进行log softmax操作。如果output_logsoftmax为真，
        # 则使用F.log_softmax函数对self.logit(output)进行log softmax操作，
        # 其中self.logit是一个线性层。如果output_logsoftmax为假，则直接使用self.logit(output)作为对数概率。
        return logprobs, state
# 最后，返回计算得到的对数概率logprobs和更新后的隐藏状态state。