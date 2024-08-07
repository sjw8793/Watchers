import math
import torch
from torch import nn, Tensor

class PositionalEncoder(nn.Module):
    def __init__(
            self,
            dropout: float=0.1,
            max_seq_len: int=5000,
            d_model: int=512,
            batch_first: bool=False
            ):
        """
        Parameters:
            dropout: The dropout rate

            max_seq_len: The maximum length of the input sequences

            d_model: The dimension of the output of sub-layers in the model 
                    (Vaswani et al, 2017)
        """
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = 1 if batch_first else 0

        position = torch.arange(max_seq_len).unsqueeze(1)
        # position > [0, 1, 2, ... max_seq_len]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, 1, d_model)
        # pe > [[[0.0 * d_model] * 1] * max_seq_len]

        # 짝수 자리(0, 2, ...)는 sin, 홀수 자리(1, 3, ...)는 cos
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # optimizer가 업데이트 하지 않지만 값은 존재하는 layer 등록
        # GPU로 보낼 수 있기 때문에 device mismatch 방지
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or [enc_seq_len, batch_size, dim_val]
        """

        x = x + self.pe[:x.size(self.x_dim)]

        return self.dropout(x)