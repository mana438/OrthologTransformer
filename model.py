import torch
import torch.nn as nn
import math

class CodonTransformer(nn.Module):
    def __init__(self, vocab_size_target, vocab_size_input, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.embedding_e = nn.Embedding(vocab_size_input, d_model)
        self.embedding_d = nn.Embedding(vocab_size_input, d_model)

        self.positional_encoding_e = nn.Embedding(1000, d_model)
        self.positional_encoding_d = nn.Embedding(1000, d_model)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,dropout=dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)
        
        self.fc_out_1 = nn.Linear(d_model, d_model)
        self.fc_out_2 = nn.Linear(d_model, int(d_model/2))
        self.fc_out_3 = nn.Linear(int(d_model/2), vocab_size_target)
        self.act = nn.ReLU()


        #if torch._C._jit_get_te_enabled():
        #    torch._C._jit_set_texpr_fuser_enabled(True)

        # self.transformer_model = nn.Transformer(d_model=d_model, nhead=nhead,num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout, norm_first=True , batch_first=True)

        self.d_model = d_model

    def forward(self, tgt, src, memory_calm=None):        
        if memory_calm  is not None:
            embedded_first = self.embedding_e(src[:, 0]).unsqueeze(1)
            # print(f"embedded_first.shape: {embedded_first.shape}")
            # print(f"memory_calm.shape: {memory_calm.shape}")
            memory = torch.cat([embedded_first, memory_calm], dim=1)
            # print(f"memory.shape after concatenation: {memory.shape}")
            memory = self.encoder(memory)
            # print(f"memory.shape after encoder: {memory.shape}")

            
            
        else:
            src = self.embedding_e(src)
            src = src + self.positional_encoding_e(torch.arange(src.size(dim=1)).unsqueeze(0).repeat(src.size(0), 1).to(src.device))
            memory = self.encoder(src)

        tgt = self.embedding_d(tgt)
        tgt = tgt + self.positional_encoding_d(torch.arange(tgt.size(dim=1)).unsqueeze(0).repeat(tgt.size(0), 1).to(tgt.device))
        tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)   
        output_codon = self.fc_out_3(self.act(self.fc_out_2(self.act(self.fc_out_1(output)))))
        
        return output_codon
