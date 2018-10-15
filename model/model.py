import torch
import torch.nn as nn
import torch.nn.functional as F

from .mynn import LSTM, Linear
import argparse






class BiDAF(nn.Module):
    def __init__(self,char_vocab_size,char_dim,char_channel_num,char_channel_width,\
                      word_vocab_size,word_dim,dropout_rate):
        super(BiDAF, self).__init__()

        self.char_vocab_size = char_vocab_size
        self.char_dim  = char_dim
        self.char_channel_num = char_channel_num
        self.char_channel_width = char_channel_width

        self.word_vocab_size = word_vocab_size
        self.word_dim = word_dim

        self.dropout_rate = dropout_rate

        # 1. Character Embedding Layer
      
        self.char_emb = nn.Embedding(char_vocab_size,char_dim, padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)
        self.char_conv = nn.Conv2d(1, char_channel_num, (char_dim, char_channel_width))

        # 2. Word Embedding Layer
        # initialize word embedding with GloVe
        # 因為之後不能用Glove 先試看看隨便塞個random的向量當作pretrained
        # TODO add pretrained embedding

        print()
        self.word_emb = nn.Embedding(word_vocab_size, word_dim, padding_idx=1)
        nn.init.uniform_( self.word_emb.weight, -0.001, 0.001)
       
    

        # highway network
        assert (char_channel_num + word_dim)%2 == 0
        hidden_dim  = (char_channel_num + word_dim)//2
        for i in range(2):
            setattr(self, f'highway_linear{i}',
                    nn.Sequential(Linear(hidden_dim * 2,hidden_dim * 2),
                                  nn.ReLU()))
            setattr(self, f'highway_gate{i}',
                    nn.Sequential(Linear(hidden_dim * 2, hidden_dim * 2),
                                  nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=hidden_dim * 2,
                                 hidden_size=hidden_dim,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=dropout_rate)

        # 4. Attention Flow Layer
        self.att_weight_c = Linear(hidden_dim * 2, 1)
        self.att_weight_q = Linear(hidden_dim * 2, 1)
        self.att_weight_cq = Linear(hidden_dim * 2, 1)

        # 5. Modeling Layer
        self.modeling_LSTM1 = LSTM(input_size=hidden_dim * 8,
                                   hidden_size=hidden_dim,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=dropout_rate)

        self.modeling_LSTM2 = LSTM(input_size=hidden_dim * 2,
                                   hidden_size=hidden_dim,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=dropout_rate)

        # 6. Output Layer
        self.p1_weight_g = Linear(hidden_dim * 8, 1, dropout=dropout_rate)
        self.p1_weight_m = Linear(hidden_dim * 2, 1, dropout=dropout_rate)
        self.p2_weight_g = Linear(hidden_dim * 8, 1, dropout=dropout_rate)
        self.p2_weight_m = Linear(hidden_dim * 2, 1, dropout=dropout_rate)

        self.output_LSTM = LSTM(input_size=hidden_dim * 2,
                                hidden_size=hidden_dim,
                                bidirectional=True,
                                batch_first=True,
                                dropout=dropout_rate)

        self.dropout = nn.Dropout(p=dropout_rate)



    def set_word_embedding(self,pretrained):
        self.embedding =  nn.Embedding.from_pretrained(pretrained, freeze=False)

    def get_hypers(self):
        return {"char_vocab_size":self.char_vocab_size,"char_dim":self.char_dim,"char_channel_num":self.char_channel_num,"char_channel_width":self.char_channel_width,\
                "word_vocab_size":self.word_vocab_size,"word_dim":self.word_dim,"dropout_rate":self.dropout_rate}



    def forward(self, batch):
        # TODO: More memory-efficient architecture
        def char_emb_layer(x):
            """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            batch_size = x.size(0)
            # (batch, seq_len, word_len, char_dim)
            x = self.char_emb(x)
            x = self.dropout(x)
            # (batch * seq_len, 1, char_dim, word_len)
            x = x.view(-1, self.char_dim, x.size(2)).unsqueeze(1)
            # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
            x = self.char_conv(x)
            x = x.squeeze(2)
            # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze(2)
            # (batch, seq_len, char_channel_size)
            x = x.view(batch_size, -1, self.char_channel_num)

            return x

        # highway_network : 為了捕捉character 和 word embedding之間的關係
        #                   highway network最後會輸出一個向量(結合character和word)
        def highway_network(x1, x2):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, f'highway_linear{i}')(x)
                g = getattr(self, f'highway_gate{i}')(x)
                x = g * h + (1 - g) * x
            # (batch, seq_len, hidden_size * 2)
            return x

        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            # (batch, c_len, q_len, hidden_size * 2)
            #c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #cq_tiled = c_tiled * q_tiled
            #cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)

            cq = []
            for i in range(q_len):
                #(batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                #(batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze(2)
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)

            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq

            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze(1)
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        def output_layer(g, m, l):
            """
            :param g: (batch, c_len, hidden_size * 8)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            # (batch, c_len)
            p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze(2)
            # (batch, c_len, hidden_size * 2)
            m2 = self.output_LSTM((m, l))[0]
            # (batch, c_len)
            p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze(2)
            return p1, p2

        # 1. Character Embedding Layer
        c_char = char_emb_layer(batch.c_char)
        q_char = char_emb_layer(batch.q_char)
        # 2. Word Embedding Layer
        c_word = self.word_emb(batch.c_word[0])
        q_word = self.word_emb(batch.q_word[0])
        c_lens = batch.c_word[1]
        q_lens = batch.q_word[1]

        # Highway network
        c = highway_network(c_char, c_word)
        q = highway_network(q_char, q_word)
        # 3. Contextual Embedding Layer
        c = self.context_LSTM((c, c_lens))[0]
        q = self.context_LSTM((q, q_lens))[0]
        # 4. Attention Flow Layer
        g = att_flow_layer(c, q)
        # 5. Modeling Layer
        m = self.modeling_LSTM2((self.modeling_LSTM1((g, c_lens))[0], c_lens))[0]
        # 6. Output Layer
        p1, p2 = output_layer(g, m, c_lens)

        # (batch, c_len), (batch, c_len)
        return p1, p2



    def dump(self,path):
        torch.save({'hypers':self.get_hypers(),"state_dict":self.state_dict()},path)


    @staticmethod
    def load_model(path):
        d = torch.load(path)
        model = BiDAF(**d['hypers'])
        model.load_state_dict(d['state_dict'])
        return model





            

