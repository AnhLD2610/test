"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        # The dimensionality of input and output is d_model = 512
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient
        
        # torch.arange(1, 4) return torch.tensor from [1,4)
        # torch.arange(1, 4) tensor([ 1,  2,  3])
        # tensor([ 1,  2,  3]) => pos = tensor chứa pos của các từ trong câu 
        pos = torch.arange(0, max_len, device=device)
        
        # tensor([[ 1],
        # [ 2],
        # [ 3]])
        # torch.Size([4, 1])
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)
        
        # encoding shape = [max_len, d_model]
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]

# d_model = 512
# max_len = 100
# device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, otherwise CPU
# positional_encoding = PositionalEncoding(d_model, max_len, device)

# # Create a sample input tensor (you can change the dimensions as needed)
# batch_size = 2
# seq_len = 30
# input_tensor = torch.randn(batch_size, seq_len)

# # Apply positional encoding
# output = positional_encoding(input_tensor)

# # Check the shape of the output
# print("Input shape:", input_tensor.shape)
# print("Output shape (positional encodings added):", output.shape)


d_model = 4
max_len = 100

# Create a sample input tensor (you can change the dimensions as needed)
batch_size = 2
seq_len = 3
input_tensor = torch.randn(batch_size, seq_len, d_model)
input_tensor1 = torch.randn(seq_len, d_model)
print(input_tensor+input_tensor1)
input = input_tensor+input_tensor1
print(input.shape)
print(input_tensor)
print(input_tensor1)