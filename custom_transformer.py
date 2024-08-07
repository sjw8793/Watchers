import math
import torch
from torch import nn, Tensor
from encoder import PositionalEncoder

class TimeSeriesTransformer(nn.Module):

    def __init__(self,
                 input_size: int,
                 dec_seq_len: int,
                 batch_first: bool,
                 out_seq_len: int=58,
                 dim_val: int=512,
                 n_encoder_layers: int=4,
                 n_decoder_layers: int=4,
                 n_heads: int=8,
                 dropout_encoder: float=0.2,
                 dropout_decoder: float=0.2,
                 dropout_pos_enc: float=0.1,
                 dim_feedforward_encoder: int=2048,
                 dim_feedforward_decoder: int=2048,
                 num_predicted_features: int=1):
        """
        Args:
            input_size: int, number of input variables. 1 if univariate.

            dec_seq_len: int, the length of the input sequence fed to the decoder
        """
        super().__init__()

        self.dec_seq_len = dec_seq_len

        # Creating the three linear layers needed for the model
        # nn.Linear: 선형 변환을 수행하는 클래스로, fully connected layer 혹은 dense layer라고도 부름
        # * in_features: 입력 텐서의 크기로, 입력 텐서의 차원 또는 feature 수를 뜻함(여기서는 전자)
        # * out_features: 출력 텐서의 크기로, 출력 텐서의 차원 또는 feature 수를 뜻함(여기서는 전자)
        # => 입력 크기는 input_size, 출력 크기는 dim_val인 선형 변환 수행
        self.encoder_input_layer = nn.Linear(
            in_features=input_size,
            out_features=dim_val
        )
        self.decoder_input_layer = nn.Linear(
            in_features=num_predicted_features,
            out_features=dim_val
        )
        # 최종 결과물을 출력하는 출력 레이어 (디코더의 맨 끝)
        self.linear_mapping = nn.Linear(
            in_features=dim_val,
            out_features=num_predicted_features
        )

        # Create Positional Encoder
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc
        )

        # The encoder layer used in the paper is identical to the one used by Vaswani et al (2017)
        # on which the PyTorch module is based.
        # nn.TransformerEncoderLayer: self-attention + add&norm + feed-forward 형태의 레이어를 생성
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
        )
        # Stack the encoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in this case, because nn.TransformerEncoderLyaer per default normalizes after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930)
        # nn.TransformerEncoder는 자동으로 레이어마다 norm을 수행하도록 설계되어 있기 때문에 옵션 파라미터 norm=None을 부여
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None
        )
        
        # 디코더 레이어 생성
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=True
        )
        # 디코더 레이어 스택
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=None
        )

    