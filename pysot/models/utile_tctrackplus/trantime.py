import copy
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import MultiheadAttention, LayerNorm, TransformerEncoderLayer, TransformerDecoderLayer
from paddle.nn import Dropout

class Cattention(nn.Layer):

    def __init__(self, in_dim):
        super(Cattention, self).__init__()
        self.chanel_in = in_dim
        self.conv1 = nn.Sequential(
            nn.Conv2DTranspose(in_dim*2, in_dim, kernel_size=1, stride=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2D(in_dim, in_dim, kernel_size=3, stride=3),
            nn.BatchNorm2D(in_dim),
            nn.ReLU(inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.linear1 = nn.Conv2D(in_dim, in_dim // 6, 1, bias_attr=False)
        self.linear2 = nn.Conv2D(in_dim // 6, in_dim, 1, bias_attr=False)
        self.gamma = paddle.create_parameter(shape=[1], dtype='float32', default_initializer=paddle.nn.initializer.Constant(0.0))
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

    def forward(self, x, y):
        ww = self.linear2(self.dropout(self.activation(self.linear1(self.avg_pool(self.conv2(y))))))
        weight = self.conv1(paddle.concat((x, y), 1)) * ww

        return x + self.gamma * weight


class Transformertime(nn.Layer):

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 384, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[nn.Layer] = None,
                 custom_decoder: Optional[nn.Layer] = None) -> None:
        super(Transformertime, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src: paddle.Tensor, srcc: paddle.Tensor, tgt: paddle.Tensor,
                src_mask: Optional[paddle.Tensor] = None, tgt_mask: Optional[paddle.Tensor] = None,
                memory_mask: Optional[paddle.Tensor] = None, src_key_padding_mask: Optional[paddle.Tensor] = None,
                tgt_key_padding_mask: Optional[paddle.Tensor] = None,
                memory_key_padding_mask: Optional[paddle.Tensor] = None) -> paddle.Tensor:

        if src.shape[1] != tgt.shape[1]:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.shape[2] != self.d_model or tgt.shape[2] != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, srcc, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, srcc, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return memory, output

    def generate_square_subsequent_mask(self, sz: int) -> paddle.Tensor:
        mask = (paddle.triu(paddle.ones([sz, sz])) == 1).transpose([0, 1])
        mask = mask.astype('float32').masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        for p in self.parameters():
            if p.ndim > 1:
                paddle.nn.initializer.XavierUniform(p)


class TransformerEncoder(nn.Layer):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: paddle.Tensor, srcc: paddle.Tensor, mask: Optional[paddle.Tensor] = None,
                src_key_padding_mask: Optional[paddle.Tensor] = None) -> paddle.Tensor:
        output = src

        for mod in self.layers:
            output = mod(output, srcc, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def _get_clones(self, module, N):
        return nn.LayerList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(nn.Layer):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = self._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: paddle.Tensor, srcc: paddle.Tensor, memory: paddle.Tensor,
                tgt_mask: Optional[paddle.Tensor] = None, memory_mask: Optional[paddle.Tensor] = None,
                tgt_key_padding_mask: Optional[paddle.Tensor] = None,
                memory_key_padding_mask: Optional[paddle.Tensor] = None) -> paddle.Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output, srcc, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def _get_clones(self, module, N):
        return nn.LayerList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer(nn.Layer):

    def __init__(self, d_model, nhead, dim_feedforward=384, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn1 = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn3 = MultiheadAttention(d_model, nhead, dropout=dropout)
        channel = dim_feedforward // 2
        self.modulation = Cattention(channel)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = Dropout(dropout)
        self.activation = self._get_activation_fn(activation)

    def forward(self, src: paddle.Tensor, srcc: paddle.Tensor, src_mask: Optional[paddle.Tensor] = None,
                src_key_padding_mask: Optional[paddle.Tensor] = None) -> paddle.Tensor:
        b, c, s = src.transpose([1, 2, 0]).shape
        src1 = self.self_attn1(srcc, src, src, attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask)[0]
        srcs1 = src + self.dropout1(src1)
        srcs1 = self.norm1(srcs1)
        src2 = self.self_attn2(srcs1, srcs1, srcs1, attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask)[0]
        srcs2 = srcs1 + self.dropout2(src2)
        srcs2 = self.norm2(srcs2)
        src = self.modulation(srcs2.reshape([b, c, int(s**0.5), int(s**0.5)]) \
                              , srcs1.contiguous().reshape([b, c, int(s**0.5), int(s**0.5)])).reshape([b, c, -1]).transpose([2, 0, 1])
        src2 = self.self_attn3(src, src, src, attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask)[0]
        srcs1 = src + self.dropout3(src2)
        srcs1 = self.norm3(srcs1)
        return srcs1

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class TransformerDecoderLayer(nn.Layer):

    def __init__(self, d_model, nhead, dim_feedforward=384, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn1 = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn2 = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.dropout4 = Dropout(dropout)
        self.activation = self._get_activation_fn(activation)

    def forward(self, tgt: paddle.Tensor, srcc: paddle.Tensor, memory: paddle.Tensor,
                tgt_mask: Optional[paddle.Tensor] = None, memory_mask: Optional[paddle.Tensor] = None,
                tgt_key_padding_mask: Optional[paddle.Tensor] = None,
                memory_key_padding_mask: Optional[paddle.Tensor] = None) -> paddle.Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt12 = self.multihead_attn1(tgt, memory, memory, attn_mask=memory_mask,
                                     key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt12)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
