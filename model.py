import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CodonTransformer(nn.Module):
    def __init__(self, vocab_size_target, vocab_size_target_amino, vocab_size_target_dna, vocab_size_input, vocab_size_input_amino, vocab_size_input_dna, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.embedding_e = nn.Embedding(vocab_size_input, d_model)
        self.embedding_e_pro = nn.Embedding(vocab_size_input_amino, d_model)
        self.embedding_e_dna = nn.Embedding(vocab_size_input_dna, int(d_model/3))

        self.embedding_d = nn.Embedding(vocab_size_input, d_model)
        self.embedding_d_pro = nn.Embedding(vocab_size_target_amino, d_model)
        self.embedding_d_dna = nn.Embedding(vocab_size_target_dna, int(d_model/3))

        self.positional_encoding_e = nn.Embedding(1000, d_model)
        self.positional_encoding_d = nn.Embedding(1000, d_model)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,dropout=dropout, batch_first=True, norm_first=True )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,dropout=dropout, batch_first=True, norm_first=True )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)
        
        self.fc_out_1 = nn.Linear(d_model, 200)
        self.act = nn.ReLU()
        self.fc_out_2 = nn.Linear(200, vocab_size_target)

        self.fc_out_1_pro = nn.Linear(d_model, 200)
        self.act_pro = nn.ReLU()
        self.fc_out_2_pro = nn.Linear(200, vocab_size_target_amino)

        self.fc_out_dna_1_1 = nn.Linear(d_model, 50)
        self.fc_out_dna_1_2 = nn.Linear(50, vocab_size_target_dna)

        self.fc_out_dna_2_1 = nn.Linear(d_model, 50)
        self.fc_out_dna_2_2 = nn.Linear(50, vocab_size_target_dna)

        self.fc_out_dna_3_1 = nn.Linear(d_model, 50)
        self.fc_out_dna_3_2 = nn.Linear(50, vocab_size_target_dna)


        # self.transformer_model = nn.Transformer(d_model=d_model, nhead=nhead,num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout, norm_first=True , batch_first=True)

        self.d_model = d_model

    def forward(self, tgt, tgt_protein, tgt_dna, src, src_protein, src_dna):
        # print("src")
        # print(src)
        # print("tgt")
        # print(tgt)
        
        src = self.embedding_e(src)
        src_protein = self.embedding_e_pro(src_protein)
        src_dna = self.embedding_e_dna(src_dna)
        src_dna = src_dna.view(src_dna.size()[0], int(src_dna.size()[1]/3) , self.d_model)

        src = src + self.positional_encoding_e(torch.arange(src.size(dim=1)).unsqueeze(0).repeat(src.size(0), 1).to(src.device))

        tgt = self.embedding_d(tgt)
        tgt_protein = self.embedding_d_pro(tgt_protein)
        tgt_dna = self.embedding_d_dna(tgt_dna)
        tgt_dna = tgt_dna.view(tgt_dna.size()[0], int(tgt_dna.size()[1]/3) , self.d_model)


        tgt = tgt + self.positional_encoding_d(torch.arange(tgt.size(dim=1)).unsqueeze(0).repeat(tgt.size(0), 1).to(tgt.device))
        tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(tgt.size(1)).to(src.device)

        # output = self.transformer_model(src, tgt, tgt_mask=tgt_mask) 
        memory = self.encoder(src)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        
        output_codon = self.fc_out_2(self.act(self.fc_out_1(output)))
        output_pro = self.fc_out_2_pro(self.act(self.fc_out_1_pro(output)))

        output_dna_1 = self.fc_out_dna_1_2(self.act(self.fc_out_dna_1_1(output)))
        output_dna_2 = self.fc_out_dna_2_2(self.act(self.fc_out_dna_2_1(output)))
        output_dna_3 = self.fc_out_dna_3_2(self.act(self.fc_out_dna_3_1(output)))
        output_dna = torch.cat((output_dna_1, output_dna_2, output_dna_3), 2).view(output_dna_1.size()[0], output_dna_1.size()[1] * 3, output_dna_1.size()[2])

        return output_codon, output_pro, output_dna, torch.cat((memory[:, :1, :], memory[:, 3:, :]), dim=1), output

    def forward_buk(self, tgt, tgt_protein, tgt_dna, src, src_protein, src_dna):
        # (N, S, E)
        src = self.embedding_e(src)
        src = src + self.positional_encoding_e(torch.arange(src.size(dim=1)).unsqueeze(0).repeat(src.size(0), 1).to(src.device))
        memory = self.encoder(src)

        tgt = self.embedding_d(tgt)
        tgt = tgt + self.positional_encoding_d(torch.arange(tgt.size(dim=1)).unsqueeze(0).repeat(tgt.size(0), 1).to(tgt.device))
        tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = self.fc_out_2(self.act(self.fc_out_1(output)))  

        return output_codon, _, _
