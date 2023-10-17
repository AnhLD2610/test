"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
from torch import nn

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embeddings import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, model, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        # d_model = 768 
        super(TransformerEmbedding, self).__init__()
        # tok_emb shape [batch_size, seq_len, d_model]
        self.tok_emb = TokenEmbedding(model)
        # pos_emb shape [seq_len, d_model]
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    # x la text
    def forward(self, x):
        # tok_emb shape [batch_size, seq_len, d_model]
        tok_emb = self.tok_emb.embedding(x)
        seq_len = tok_emb.shape[1]
        # pos_emb shape [seq_len, d_model]
        pos_emb = self.pos_emb(seq_len)
        # tok_emb + pos_emb shape = [batch_size, seq_len, d_model]
        return self.drop_out(tok_emb + pos_emb)
