import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.transforms import Normalize
from collections import namedtuple
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Tuple

def correct(output,target,std_ret,mean_ret):
    o=output.flatten()*std_ret +mean_ret
    t=target*std_ret + mean_ret
    tmp=o*t
    correct=0
    for i in range(len(tmp)):
        if(tmp[i].item() > 0):
            correct+=1
    return correct

def hit_profit(output,target,std_ret,mean_ret):
    o=output.flatten()*std_ret +mean_ret
    t=target*std_ret + mean_ret
    tmp=o*t
    hit_profit=0
    for i in range(len(tmp)):
        if(tmp[i].item() > 0):
            hit_profit+=torch.abs(target[i])
    return hit_profit

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    return torch.tensor(indexes, dtype=torch.long).view(-1)


def return_fun(data,delta_t,index):
    data1=data.to_numpy()
    ret_fun=[]
    for i in range(len(data1)-delta_t):
        ret_fun.append(100*(data1[i+delta_t,index]-data1[i,index])/data1[i,index])
    for i in range(delta_t):
        ret_fun.append(None)
    ret_fun=ret_fun
    return ret_fun


class LSTMNumeric(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.lstm=torch.nn.LSTM(input_size,hidden_size,num_layers=3,batch_first=True)
        self.fully_connected=torch.nn.Sequential(
            torch.nn.Linear(hidden_size*5,150),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(150,50),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(50,1)
        )
    def forward(self,x):
        batch_size=x.shape[0]
        hidden_size=30
        h0=torch.zeros(3,batch_size,hidden_size).cuda()
        c0=torch.zeros(3,batch_size,hidden_size).cuda()

        lstm_output, cels = self.lstm(x,(h0,c0))
        flatten_output=torch.flatten(lstm_output,start_dim=1)
        
        output=self.fully_connected(flatten_output)
        
        return output
    
    
class TextOnlyGRU(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,output_size):
        super(TextOnlyGRU,self).__init__()
        
        self.embedding=nn.Embedding(vocab_size,embedding_size)
        self.gru=nn.GRU(embedding_size, hidden_size,2, batch_first=True)
        self.fully_connected=nn.Sequential(
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,32),
            nn.Tanh(),
            nn.Linear(32,1)
        )
    def forward(self,x):
        embed=self.embedding(x)
        gru_output, h=self.gru(embed)
        g=gru_output[:,-1,:]
        return self.fully_connected(g)
    
    
class TextAndNumeric(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,numeric_size):
        super(TextAndNumeric,self).__init__()
        
        self.embedding=nn.Embedding(vocab_size,embedding_size)
        self.rnn_text=nn.GRU(embedding_size,hidden_size, 2 , batch_first=True)
        self.rnn_numeric=nn.LSTM(numeric_size, hidden_size, 2 , batch_first=True, bidirectional=True)
        self.numeric_fc=nn.Linear(10*hidden_size,hidden_size)
        self.fully_connected=nn.Sequential(
            nn.Linear(2*hidden_size,64),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(64,32),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(32,1)
        )
        
    def forward(self,headline,numeric):
        #text
        embed=self.embedding(headline)
        gru_output, h=self.rnn_text(embed)
        text_output=gru_output[:,-1,:]
        
        #numeric
        lstm_output, h = self.rnn_numeric(numeric)
        flatten_output=torch.flatten(lstm_output,start_dim=1)
        numeric_output=self.numeric_fc(flatten_output)
        
        #concatenate text and numeric
        hidden=torch.cat([text_output,numeric_output],1)
        output=self.fully_connected(hidden)
        
        return output
    
    
def test_tan(model,loader):
    loss_fn=torch.nn.MSELoss()

    tan.eval()
    with torch.no_grad():
        val_loss_buffer=[]
        val_ac=0
        cor=0
        hit_profit=0
        for i, (x,y) in enumerate(loader):
            headline,numeric = x
            numeric=numeric.float()
            headline=headline.cuda()
            numeric=numeric.cuda()
            y=y.cuda()
            
            output=model(headline,numeric)
            output.float()
            y.float()
            val_loss_buffer.append(loss_fn(output,y).item())
            if(correct(output, y, std_ret, mean_ret)==1):
                hit_profit += correct(output, y, std_ret, mean_ret) * abs(y*std_ret - mean_ret)
            else:
                hit_profit-= abs(y*std_ret - mean_ret)
            
            cor+=correct(output,y,std_ret,mean_ret)
            
        val_acc=cor/len(loader)
        val_loss=torch.mean(torch.tensor(val_loss_buffer))
        print(f"Test loss: {val_loss:.3f} Test Acc {val_acc:.5f}")
        print(f"Test Hit Profit: {hit_profit.item():0.3f}")
        
        return val_loss,val_acc,hit_profit
    
def test_numeric(model,loader):
    loss_fn=torch.nn.MSELoss()

    model.eval()
    with torch.no_grad():
        val_loss_buffer=[]
        val_ac=0
        cor=0
        hit_profit=0
        for i, (x,y) in enumerate(loader):
            headline,numeric = x
            numeric=numeric.float()
            numeric=numeric.cuda()
            y=y.cuda()
            
            output=model(numeric)
            output.float()
            y.float()
            val_loss_buffer.append(loss_fn(output,y).item())
            if(correct(output, y, std_ret, mean_ret)==1):
                hit_profit += correct(output, y, std_ret, mean_ret) * abs(y*std_ret - mean_ret)
            else:
                hit_profit-= abs(y*std_ret - mean_ret)
            
            cor+=correct(output,y,std_ret,mean_ret)
            
        val_acc=cor/len(loader)
        val_loss=torch.mean(torch.tensor(val_loss_buffer))
        print(f"Test loss: {val_loss:.3f} Test Acc {val_acc:.5f}")
        print(f"Test Hit Profit: {hit_profit.item():0.3f}")
        
        return val_loss,val_acc,hit_profit
    
def test_text(model,loader):
    loss_fn=torch.nn.MSELoss()

    model.eval()
    with torch.no_grad():
        val_loss_buffer=[]
        val_ac=0
        cor=0
        hit_profit=0
        for i, (x,y) in enumerate(loader):
            headline,numeric = x
            headline=headline.cuda()
            y=y.cuda()
            
            output=model(headline)
            output.float()
            y.float()
            val_loss_buffer.append(loss_fn(output,y).item())
            if(correct(output, y, std_ret, mean_ret)==1):
                hit_profit += correct(output, y, std_ret, mean_ret) * abs(y*std_ret - mean_ret)
            else:
                hit_profit-= abs(y*std_ret - mean_ret)
            
            cor+=correct(output,y,std_ret,mean_ret)
            
        val_acc=cor/len(loader)
        val_loss=torch.mean(torch.tensor(val_loss_buffer))
        print(f"Test loss: {val_loss:.3f} Test Acc {val_acc:.5f}")
        print(f"Test Hit Profit: {hit_profit.item():0.3f}")
        
        return val_loss,val_acc,hit_profit


# Transofrmer model is here bois
import torch
import torch.nn as nn
import math
from torch import nn, Tensor


class PositionalEncoder(nn.Module):
    """
    The authors of the original transformer paper describe very succinctly what
    the positional encoding layer does and why it is needed:

    "Since our model contains no recurrence and no convolution, in order for the
    model to make use of the order of the sequence, we must inject some
    information about the relative or absolute position of the tokens in the
    sequence." (Vaswani et al, 2017)

    Adapted from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(
            self,
            dropout: float = 0.1,
            max_seq_len: int = 14,
            d_model: int = 512,
            batch_first: bool = False
    ):
        """
        Parameters:

            dropout: the dropout rate

            max_seq_len: the maximum length of the input sequences

            d_model: The dimension of the output of sub-layers in the model
                     (Vaswani et al, 2017)
        """

        super().__init__()

        self.d_model = d_model

        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        self.x_dim = 1 if batch_first else 0

        # copy pasted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_len, 1, d_model)

        pe[:, 0, 0::2] = torch.sin(position * div_term)

        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or
               [enc_seq_len, batch_size, dim_val]
        """

        x = x + self.pe[:x.size(self.x_dim)]

        return self.dropout(x)


import torch.nn as nn
from torch import nn, Tensor
# import positional_encoder as pe
import torch.nn.functional as F


class TimeSeriesTransformer(nn.Module):
    """
    This class implements a transformer model that can be used for times series
    forecasting. This time series transformer model is based on the paper by
    Wu et al (2020) [1]. The paper will be referred to as "the paper".

    A detailed description of the code can be found in my article here:

    https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e

    In cases where the paper does not specify what value was used for a specific
    configuration/hyperparameter, this class uses the values from Vaswani et al
    (2017) [2] or from PyTorch source code.

    Unlike the paper, this class assumes that input layers, positional encoding
    layers and linear mapping layers are separate from the encoder and decoder,
    i.e. the encoder and decoder only do what is depicted as their sub-layers
    in the paper. For practical purposes, this assumption does not make a
    difference - it merely means that the linear and positional encoding layers
    are implemented inside the present class and not inside the
    Encoder() and Decoder() classes.

    [1] Wu, N., Green, B., Ben, X., O'banion, S. (2020).
    'Deep Transformer Models for Time Series Forecasting:
    The Influenza Prevalence Case'.
    arXiv:2001.08317 [cs, stat] [Preprint].
    Available at: http://arxiv.org/abs/2001.08317 (Accessed: 9 March 2022).

    [2] Vaswani, A. et al. (2017)
    'Attention Is All You Need'.
    arXiv:1706.03762 [cs] [Preprint].
    Available at: http://arxiv.org/abs/1706.03762 (Accessed: 9 March 2022).

    """

    def __init__(self,
                 input_size: int,
                 dec_seq_len: int,
                 batch_first: bool,
                 out_seq_len: int = 58,
                 dim_val: int = 512,
                 n_encoder_layers: int = 4,
                 n_decoder_layers: int = 4,
                 n_heads: int = 8,
                 dropout_encoder: float = 0.2,
                 dropout_decoder: float = 0.2,
                 dropout_pos_enc: float = 0.1,
                 dim_feedforward_encoder: int = 2048,
                 dim_feedforward_decoder: int = 2048,
                 num_predicted_features: int = 1
                 ):
        """
        Args:

            input_size: int, number of input variables. 1 if univariate.

            dec_seq_len: int, the length of the input sequence fed to the decoder

            dim_val: int, aka d_model. All sub-layers in the model produce
                     outputs of dimension dim_val

            n_encoder_layers: int, number of stacked encoder layers in the encoder

            n_decoder_layers: int, number of stacked encoder layers in the decoder

            n_heads: int, the number of attention heads (aka parallel attention layers)

            dropout_encoder: float, the dropout rate of the encoder

            dropout_decoder: float, the dropout rate of the decoder

            dropout_pos_enc: float, the dropout rate of the positional encoder

            dim_feedforward_encoder: int, number of neurons in the linear layer
                                     of the encoder

            dim_feedforward_decoder: int, number of neurons in the linear layer
                                     of the decoder

            num_predicted_features: int, the number of features you want to predict.
                                    Most of the time, this will be 1 because we're
                                    only forecasting FCR-N prices in DK2, but in
                                    we wanted to also predict FCR-D with the same
                                    model, num_predicted_features should be 2.
        """

        super().__init__()

        self.dec_seq_len = dec_seq_len

        # print("input_size is: {}".format(input_size))
        # print("dim_val is: {}".format(dim_val))

        # Creating the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(
            in_features=input_size,
            out_features=dim_val
        )

        self.decoder_input_layer = nn.Linear(
            in_features=num_predicted_features,
            out_features=dim_val
        )

        self.linear_mapping = nn.Linear(
            in_features=dim_val,
            out_features=num_predicted_features
        )

        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc
        )

        # The encoder layer used in the paper is identical to the one used by
        # Vaswani et al (2017) on which the PyTorch module is based.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
        )

        # Stack the encoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerEncoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first
        )

        # Stack the decoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerDecoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=None
        )

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor = None,
                tgt_mask: Tensor = None) -> Tensor:
        """
        Returns a tensor of shape:

        [target_sequence_length, batch_size, num_predicted_features]

        Args:

            src: the encoder's output sequence. Shape: (S,E) for unbatched input,
                 (S, N, E) if batch_first=False or (N, S, E) if
                 batch_first=True, where S is the source sequence length,
                 N is the batch size, and E is the number of features (1 if univariate)

            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input,
                 (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if
                 batch_first=True, where T is the target sequence length,
                 N is the batch size, and E is the number of features (1 if univariate)

            src_mask: the mask for the src sequence to prevent the model from
                      using data points from the target sequence

            tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence


        """

        # print("From model.forward(): Size of src as given to forward(): {}".format(src.size()))
        # print("From model.forward(): tgt size = {}".format(tgt.size()))

        # Pass throguh the input layer right before the encoder
        src = src.to(torch.float32)
        src = self.encoder_input_layer(src)  # src shape: [batch_size, src length, dim_val] regardless of number of input features
        # print("From model.forward(): Size of src after input layer: {}".format(src.size()))

        # Pass through the positional encoding layer
        src = self.positional_encoding_layer(
            src)  # src shape: [batch_size, src length, dim_val] regardless of number of input features
        # print("From model.forward(): Size of src after pos_enc layer: {}".format(src.size()))

        # Pass through all the stacked encoder layers in the encoder
        # Masking is only needed in the encoder if input sequences are padded
        # which they are not in this time series use case, because all my
        # input sequences are naturally of the same length.
        # (https://github.com/huggingface/transformers/issues/4083)
        src = self.encoder(  # src shape: [batch_size, enc_seq_len, dim_val]
            src=src
        )
        # print("From model.forward(): Size of src after encoder: {}".format(src.size()))

        # Pass decoder input through decoder input layer
        tgt = tgt.to(torch.float32)
        decoder_output = self.decoder_input_layer(
            tgt)  # src shape: [target sequence length, batch_size, dim_val] regardless of number of input features
        # print("From model.forward(): Size of decoder_output after linear decoder layer: {}".format(decoder_output.size()))

        # if src_mask is not None:
        #     print("From model.forward(): Size of src_mask: {}".format(src_mask.size()))
        # if tgt_mask is not None:
        #     print("From model.forward(): Size of tgt_mask: {}".format(tgt_mask.size()))

        # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
        )

        # print("From model.forward(): decoder_output shape after decoder: {}".format(decoder_output.shape))

        # Pass through linear mapping
        decoder_output = self.linear_mapping(decoder_output)  # shape [batch_size, target seq len]
        # print("From model.forward(): decoder_output size after linear_mapping = {}".format(decoder_output.size()))

        return decoder_output

    def get_src_trg(
            self,
            sequence: torch.Tensor,
            enc_seq_len: int,
            target_seq_len: int
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Generate the src (encoder input), trg (decoder input) and trg_y (the target)
        sequences from a sequence.
        Args:
            sequence: tensor, a 1D tensor of length n where
                    n = encoder input length + target sequence length
            enc_seq_len: int, the desired length of the input to the transformer encoder
            target_seq_len: int, the desired length of the target sequence (the
                            one against which the model output is compared)
        Return:
            src: tensor, 1D, used as input to the transformer model
            trg: tensor, 1D, used as input to the transformer model
            trg_y: tensor, 1D, the target sequence against which the model output
                is compared when computing loss.

        """
        # print("Called dataset.TransformerDataset.get_src_trg")
        assert len(
            sequence) == enc_seq_len + target_seq_len, "Sequence length does not equal (input length + target length)"

        # print("From data.TransformerDataset.get_src_trg: sequence shape: {}".format(sequence.shape))

        # encoder input
        src = sequence[:enc_seq_len]

        # decoder input. As per the paper, it must have the same dimension as the
        # target sequence, and it must contain the last value of src, and all
        # values of trg_y except the last (i.e. it must be shifted right by 1)
        trg = sequence[enc_seq_len - 1:len(sequence) - 1]

        # print("From data.TransformerDataset.get_src_trg: trg shape before slice: {}".format(trg.shape))

        trg = trg[:, 0]

        # print("From data.TransformerDataset.get_src_trg: trg shape after slice: {}".format(trg.shape))

        if len(trg.shape) == 1:
            trg = trg.unsqueeze(-1)

            # print("From data.TransformerDataset.get_src_trg: trg shape after unsqueeze: {}".format(trg.shape))

        assert len(trg) == target_seq_len, "Length of trg does not match target sequence length"

        # The target sequence against which the model output will be compared to compute loss
        trg_y = sequence[-target_seq_len:]

        # print("From data.TransformerDataset.get_src_trg: trg_y shape before slice: {}".format(trg_y.shape))

        # We only want trg_y to consist of the target variable not any potential exogenous variables
        trg_y = trg_y[:, 0]

        # print("From data.TransformerDataset.get_src_trg: trg_y shape after slice: {}".format(trg_y.shape))

        assert len(trg_y) == target_seq_len, "Length of trg_y does not match target sequence length"

        return src, trg, trg_y.squeeze(
            -1)  # change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len]
