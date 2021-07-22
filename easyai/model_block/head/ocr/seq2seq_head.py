#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:lipeijie

from easyai.name_manager.block_name import HeadType
from easyai.model_block.base_block.transformer.transformer_encoder import TransformerEncoder
from easyai.model_block.base_block.transformer.transformer_decoder import TransformerDecoder
from easyai.model_block.utility.base_block import *


class Seq2SeqHead(BaseBlock):

    def __init__(self, in_channels, num_classes, ignore_index=2):
        super().__init__(HeadType.Seq2SeqHead)
        self.trg_pad_idx = ignore_index
        self.encoder = TransformerEncoder(in_channels, 3, 8, 512)
        self.decoder = TransformerDecoder(num_classes, in_channels, 3, 8, 512)

    def make_src_mask(self, src):

        # src = [batch size, src len]

        # src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return None

    def make_trg_mask(self, trg):

        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # import pdb;pdb.set_trace()
        # print("trg_pad_mask: ", trg_pad_mask)
        trg_pad_mask[:, :, :, 0] = 1
        # print("trg_pad_mask: ", trg_pad_mask)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):
        # src = [batch size, src len]
        # trg = [batch size, trg len]
        batch_size = src.shape[0]
        if self.training:
            src_mask = None  # self.make_src_mask(src)
            trg_mask = self.make_trg_mask(trg)
            # import pdb;pdb.set_trace()
            # print(trg)
            # print(mmm)

            # src_mask = [batch size, 1, 1, src len]
            # trg_mask = [batch size, 1, trg len, trg len]

            enc_src = self.encoder(src, src_mask)

            # enc_src = [batch size, src len, hid dim]

            output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

            # output = [batch size, trg len, output dim]
            # attention = [batch size, n heads, trg len, src len]
        else:
            # tokens = [src_field.init_token] + tokens + [src_field.eos_token]

            # src_indexes = [src_field.vocab.stoi[token] for token in tokens]

            # src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

            # src_mask = model.make_src_mask(src_tensor)

            # with torch.no_grad():
            max_len = trg.shape[1] - 1
            # targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
            # probs = torch.FloatTensor(batch_size, max_len, self.num_classes).fill_(0).to(device)
            src_mask = None
            enc_src = self.encoder(src, src_mask)

            # trg_indexes = [0]
            trg_tensor = trg

            for i in range(max_len):
                # trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

                trg_mask = self.make_trg_mask(trg_tensor)
                # import pdb; pdb.set_trace()

                # with torch.no_grad():
                output, attention = self.decoder(trg_tensor, enc_src, trg_mask, src_mask)

                pred_token = output.argmax(2)[:, i]
                trg_tensor[:, i + 1] = pred_token

                # trg_indexes.append(pred_token)

                # if pred_token == :
                #    break

            # trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
            # print(trg_tensor)

        return output
