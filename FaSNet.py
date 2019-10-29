import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

from utility.models import *

# DPRNN for beamforming filter estimation
class BF_module(DPRNN_base):
    def __init__(self, *args, **kwargs):
        super(BF_module, self).__init__(*args, **kwargs)
        
        # gated output layer
        self.output = nn.Sequential(nn.Conv1d(self.feature_dim, self.output_dim, 1),
                                    nn.Tanh()
                                   )
        self.output_gate = nn.Sequential(nn.Conv1d(self.feature_dim, self.output_dim, 1),
                                         nn.Sigmoid()
                                        )
        
    def forward(self, input, num_mic):
        
        if self.model_type == 'DPRNN':
            # input: (B, N, T)
            batch_size, N, seq_length = input.shape
            ch = 1
        elif self.model_type == 'DPRNN_TAC':
            # input: (B, ch, N, T)
            batch_size, ch, N, seq_length = input.shape

        input = input.view(batch_size*ch, N, seq_length)  # B*ch, N, T
        enc_feature = self.BN(input)
        
        # split the encoder output into overlapped, longer segments
        enc_segments, enc_rest = self.split_feature(enc_feature, self.segment_size)  # B*ch, N, L, K
        
        # pass to DPRNN
        if self.model_type == 'DPRNN':
            output = self.DPRNN(enc_segments).view(batch_size*ch*self.num_spk, self.feature_dim, self.segment_size, -1)  # B*ch*nspk, N, L, K
        elif self.model_type == 'DPRNN_TAC':
            enc_segments = enc_segments.view(batch_size, ch, -1, enc_segments.shape[2], enc_segments.shape[3])  # B, ch, N, L, K
            output = self.DPRNN(enc_segments, num_mic).view(batch_size*ch*self.num_spk, self.feature_dim, self.segment_size, -1)  # B*ch*nspk, N, L, K
        
        # overlap-and-add of the outputs
        output = self.merge_feature(output, enc_rest)  # B*ch*nspk, N, T

        # gated output layer for filter generation
        bf_filter = self.output(output) * self.output_gate(output)  # B*ch*nspk, K, T
        bf_filter = bf_filter.transpose(1, 2).contiguous().view(batch_size, ch, self.num_spk, -1, self.output_dim)  # B, ch, nspk, L, N
        
        return bf_filter


# base module for FaSNet
class FaSNet_base(nn.Module):
    def __init__(self, enc_dim, feature_dim, hidden_dim, layer, segment_size=50, 
                 nspk=2, win_len=4, context_len=16, sr=16000):
        super(FaSNet_base, self).__init__()

        # parameters
        self.window = int(sr * win_len / 1000)
        self.context = int(sr * context_len / 1000)
        self.stride = self.window // 2
        
        self.filter_dim = self.context*2+1
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.segment_size = segment_size
        
        self.layer = layer
        self.num_spk = nspk
        self.eps = 1e-8

        # waveform encoder
        self.encoder = nn.Conv1d(1, self.enc_dim, self.context*2+self.window, bias=False)
        self.enc_LN = nn.GroupNorm(1, self.enc_dim, eps=1e-8)

    def pad_input(self, input, window):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nmic, nsample = input.shape
        stride = window // 2

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, nmic, rest).type(input.type())
            input = torch.cat([input, pad], 2)
        pad_aux = torch.zeros(batch_size, nmic, stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest


    def seg_signal_context(self, x, window, context):
        """
        Segmenting the signal into chunks with specific context.
        input:
            x: size (B, ch, T)
            window: int
            context: int

        """
        
        # pad input accordingly
        # first pad according to window size
        input, rest = self.pad_input(x, window)
        batch_size, nmic, nsample = input.shape
        stride = window // 2
        
        # pad another context size
        pad_context = torch.zeros(batch_size, nmic, context).type(input.type())
        input = torch.cat([pad_context, input, pad_context], 2)  # B, ch, L
        
        # calculate index for each chunk
        nchunk = 2*nsample // window - 1
        begin_idx = np.arange(nchunk)*stride
        begin_idx = torch.from_numpy(begin_idx).type(input.type()).long().view(1, 1, -1)  # 1, 1, nchunk
        begin_idx = begin_idx.expand(batch_size, nmic, nchunk)  # B, ch, nchunk
        # select entries from index
        chunks = [torch.gather(input, 2, begin_idx+i).unsqueeze(3) for i in range(2*context + window)]  # B, ch, nchunk, 1
        chunks = torch.cat(chunks, 3)  # B, ch, nchunk, chunk_size
        
        # center frame
        center_frame = chunks[:,:,:,context:context+window]

        return center_frame, chunks, rest

    def seq_cos_sim(self, ref, target):
        """
        Cosine similarity between some reference mics and some target mics
        ref: shape (nmic1, L, seg1)
        target: shape (nmic2, L, seg2)
        """
        
        assert ref.size(1) == target.size(1), "Inputs should have same length."
        assert ref.size(2) >= target.size(2), "Reference input should be no smaller than the target input."
        
        seq_length = ref.size(1)
        
        larger_ch = ref.size(0)
        if target.size(0) > ref.size(0):
            ref = ref.expand(target.size(0), ref.size(1), ref.size(2)).contiguous()  # nmic2, L, seg1
            larger_ch = target.size(0)
        elif target.size(0) < ref.size(0):
            target = target.expand(ref.size(0), target.size(1), target.size(2)).contiguous()  # nmic1, L, seg2
        
        # L2 norms
        ref_norm = F.conv1d(ref.view(1, -1, ref.size(2)).pow(2), 
                            torch.ones(ref.size(0)*ref.size(1), 1, target.size(2)).type(ref.type()),
                            groups=larger_ch*seq_length)  # 1, larger_ch*L, seg1-seg2+1
        ref_norm = ref_norm.sqrt() + self.eps
        target_norm = target.norm(2, dim=2).view(1, -1, 1) + self.eps  # 1, larger_ch*L, 1
        # cosine similarity
        cos_sim = F.conv1d(ref.view(1, -1, ref.size(2)),
                           target.view(-1, 1, target.size(2)),
                           groups=larger_ch*seq_length)  # 1, larger_ch*L, seg1-seg2+1
        cos_sim = cos_sim / (ref_norm * target_norm)

        return cos_sim.view(larger_ch, seq_length, -1)

    def forward(self, input, num_mic):
        """
        input: shape (batch, max_num_ch, T)
        num_mic: shape (batch, ), the number of channels for each input. Zero for fixed geometry configuration.
        """
        pass
        
    

# original FaSNet
class FaSNet_origin(FaSNet_base):
    def __init__(self, *args, **kwargs):
        super(FaSNet_origin, self).__init__(*args, **kwargs)
        
        # DPRNN for ref mic
        self.ref_BF = BF_module(self.filter_dim+self.enc_dim, self.feature_dim, self.hidden_dim, 
                                self.filter_dim, self.num_spk, self.layer, self.segment_size, model_type='DPRNN')
                    
        # DPRNN for other mics
        self.other_BF = BF_module(self.filter_dim+self.enc_dim, self.feature_dim, self.hidden_dim,
                                  self.filter_dim, 1, self.layer, self.segment_size, model_type='DPRNN')
        
        
    def forward(self, input, num_mic):

        batch_size = input.size(0)
        nmic = input.size(1)
        
        # split input into chunks
        all_seg, all_mic_context, rest = self.seg_signal_context(input, self.window, self.context)  # B, nmic, L, win/chunk
        seq_length = all_seg.size(2)
        
        # first step: filtering the ref mic to create a clean estimate
        # calculate cosine similarity
        ref_context = all_mic_context[:,0].contiguous().view(1, -1, self.context*2+self.window)  # 1, B*L, 3*win
        other_segment = all_seg[:,1:].contiguous().transpose(0, 1).contiguous().view(nmic-1, -1, self.window) # nmic-1, B*L, win
        ref_cos_sim = self.seq_cos_sim(ref_context, other_segment)  # nmic-1, B*L, 2*win+1
        ref_cos_sim = ref_cos_sim.view(nmic-1, batch_size, seq_length, self.filter_dim)  # nmic-1, B, L, 2*win+1
        if num_mic.max() == 0:
            ref_cos_sim = ref_cos_sim.mean(0)  # B, L, 2*win+1
            ref_cos_sim = ref_cos_sim.transpose(1, 2).contiguous()  # B, 2*win+1, L
        else:
            # consider only the valid channels
            ref_cos_sim = [ref_cos_sim[:num_mic[b],b,:].mean(0).unsqueeze(0) for b in range(batch_size)]  # 1, L, 2*win+1
            ref_cos_sim = torch.cat(ref_cos_sim, 0).transpose(1, 2).contiguous()  # B, 2*win+1, L
            
        
        # pass to a DPRNN
        ref_feature = all_mic_context[:,0].contiguous().view(batch_size*seq_length, 1, self.context*2+self.window)
        ref_feature = self.encoder(ref_feature)  # B*L, N, 1
        ref_feature = ref_feature.view(batch_size, seq_length, self.enc_dim).transpose(1, 2).contiguous()  # B, N, L
        ref_filter = self.ref_BF(torch.cat([self.enc_LN(ref_feature), ref_cos_sim], 1), num_mic)  # B, 1, nspk, L, 2*win+1
        
        # convolve with ref mic context segments
        ref_context = torch.cat([all_mic_context[:,0].unsqueeze(1)]*self.num_spk, 1)  # B, nspk, L, 3*win
        ref_output = F.conv1d(ref_context.view(1, -1, self.context*2+self.window), 
                              ref_filter.view(-1, 1, self.filter_dim),
                              groups=batch_size*self.num_spk*seq_length) # 1, B*nspk*L, win
        ref_output = ref_output.view(batch_size*self.num_spk, seq_length, self.window)  # B*nspk, L, win
        
        # second step: use the ref output as the cue, beamform other mics
        # calculate cosine similarity
        other_context = torch.cat([all_mic_context[:,1:].unsqueeze(1)]*self.num_spk, 1)  # B, nspk, nmic-1, L, 3*win
        other_context_saved = other_context.view(batch_size*self.num_spk, nmic-1, seq_length, self.context*2+self.window)  # B*nspk, nmic-1, L, 3*win
        other_context = other_context_saved.transpose(0, 1).contiguous().view(nmic-1, -1, self.context*2+self.window)  # nmic-1, B*nspk*L, 3*win
        ref_segment = ref_output.view(1, -1, self.window)  # 1, B*nspk*L, win
        other_cos_sim = self.seq_cos_sim(other_context, ref_segment)  # nmic-1, B*nspk*L, 2*win+1
        other_cos_sim = other_cos_sim.view(nmic-1, batch_size*self.num_spk, seq_length, self.filter_dim)  # nmic-1, B*nspk, L, 2*win+1
        other_cos_sim = other_cos_sim.permute(1,0,3,2).contiguous().view(-1, self.filter_dim, seq_length)  # B*nspk*(nmic-1), 2*win+1, L
        
        # pass to another DPRNN
        other_feature = self.encoder(other_context_saved.view(-1, 1, self.context*2+self.window)).view(-1, seq_length, self.enc_dim)  # B*nspk*(nmic-1), L, N
        other_feature = other_feature.transpose(1, 2).contiguous()  # B*nspk*(nmic-1), N, L
        other_filter = self.other_BF(torch.cat([self.enc_LN(other_feature), other_cos_sim], 1), num_mic)  # B*nspk*(nmic-1), 1, 1, L, 2*win+1
        
        # convolve with other mic context segments
        other_output = F.conv1d(other_context_saved.view(1, -1, self.context*2+self.window), 
                                other_filter.view(-1, 1, self.filter_dim),
                                groups=batch_size*self.num_spk*(nmic-1)*seq_length) # 1, B*nspk*(nmic-1)*L, win
        other_output = other_output.view(batch_size*self.num_spk, nmic-1, seq_length, self.window)  # B*nspk, nmic-1, L, win
        
        all_bf_output = torch.cat([ref_output.unsqueeze(1), other_output], 1)  # B*nspk, nmic, L, win
        
        # reshape to utterance
        bf_signal = all_bf_output.view(batch_size*self.num_spk*nmic, -1, self.window*2)
        bf_signal1 = bf_signal[:,:,:self.window].contiguous().view(batch_size*self.num_spk*nmic, 1, -1)[:,:,self.stride:]
        bf_signal2 = bf_signal[:,:,self.window:].contiguous().view(batch_size*self.num_spk*nmic, 1, -1)[:,:,:-self.stride]
        bf_signal = bf_signal1 + bf_signal2  # B*nspk*nmic, 1, T
        if rest > 0:
            bf_signal = bf_signal[:,:,:-rest]

        bf_signal = bf_signal.view(batch_size, self.num_spk, nmic, -1)  # B, nspk, nmic, T
        # consider only the valid channels
        if num_mic.max() == 0:
            bf_signal = bf_signal.mean(2)  # B, nspk, T
        else:
            bf_signal = [bf_signal[b,:,:num_mic[b]].mean(1).unsqueeze(0) for b in range(batch_size)]  # nspk, T
            bf_signal = torch.cat(bf_signal, 0)  # B, nspk, T

        return bf_signal

# single-stage FaSNet + TAC
class FaSNet_TAC(FaSNet_base):
    def __init__(self, *args, **kwargs):
        super(FaSNet_TAC, self).__init__(*args, **kwargs)
        
        # DPRNN + TAC for estimation
        self.all_BF = BF_module(self.filter_dim+self.enc_dim, self.feature_dim, self.hidden_dim, 
                                self.filter_dim, self.num_spk, self.layer, self.segment_size, model_type='DPRNN_TAC')
        
    def forward(self, input, num_mic):
        
        batch_size = input.size(0)
        nmic = input.size(1)
        
        # split input into chunks
        all_seg, all_mic_context, rest = self.seg_signal_context(input, self.window, self.context)  # B, nmic, L, win/chunk
        seq_length = all_seg.size(2)
        
        # embeddings for all channels
        enc_output = self.encoder(all_mic_context.view(-1, 1, self.context*2+self.window)).view(batch_size*nmic, seq_length, self.enc_dim).transpose(1, 2).contiguous()  # B*nmic, N, L
        enc_output = self.enc_LN(enc_output).view(batch_size, nmic, self.enc_dim, seq_length)  # B, nmic, N, L
        
        # calculate the cosine similarities for ref channel's center frame with all channels' context
        
        ref_seg = all_seg[:,0].contiguous().view(1, -1, self.window)  # 1, B*L, win
        all_context = all_mic_context.transpose(0, 1).contiguous().view(nmic, -1, self.context*2+self.window)  # 1, B*L, 3*win
        all_cos_sim = self.seq_cos_sim(all_context, ref_seg)  # nmic, B*L, 2*win+1
        all_cos_sim = all_cos_sim.view(nmic, batch_size, seq_length, self.filter_dim).permute(1,0,3,2).contiguous()  # B, nmic, 2*win+1, L
        
        input_feature = torch.cat([enc_output, all_cos_sim], 2)  # B, nmic, N+2*win+1, L
        
        # pass to DPRNN
        all_filter = self.all_BF(input_feature, num_mic)  # B, ch, nspk, L, 2*win+1
        
        # convolve with all mic's context
        mic_context = torch.cat([all_mic_context.view(batch_size*nmic, 1, seq_length,
                                                      self.context*2+self.window)]*self.num_spk, 1)  # B*nmic, nspk, L, 3*win
        all_bf_output = F.conv1d(mic_context.view(1, -1, self.context*2+self.window), 
                                 all_filter.view(-1, 1, self.filter_dim),
                                 groups=batch_size*nmic*self.num_spk*seq_length) # 1, B*nmic*nspk*L, win
        all_bf_output = all_bf_output.view(batch_size, nmic, self.num_spk, seq_length, self.window)  # B, nmic, nspk, L, win

        # reshape to utterance
        bf_signal = all_bf_output.view(batch_size*nmic*self.num_spk, -1, self.window*2)
        bf_signal1 = bf_signal[:,:,:self.window].contiguous().view(batch_size*nmic*self.num_spk, 1, -1)[:,:,self.stride:]
        bf_signal2 = bf_signal[:,:,self.window:].contiguous().view(batch_size*nmic*self.num_spk, 1, -1)[:,:,:-self.stride]
        bf_signal = bf_signal1 + bf_signal2  # B*nmic*nspk, 1, T
        if rest > 0:
            bf_signal = bf_signal[:,:,:-rest]

        bf_signal = bf_signal.view(batch_size, nmic, self.num_spk, -1)  # B, nmic, nspk, T
        # consider only the valid channels
        if num_mic.max() == 0:
            bf_signal = bf_signal.mean(1)  # B, nspk, T
        else:
            bf_signal = [bf_signal[b,:num_mic[b]].mean(0).unsqueeze(0) for b in range(batch_size)]  # nspk, T
            bf_signal = torch.cat(bf_signal, 0)  # B, nspk, T

        return bf_signal


def test_model(model):
    x = torch.rand(2, 4, 32000)  # (batch, num_mic, length)
    num_mic = torch.from_numpy(np.array([3, 2])).view(-1,).type(x.type())  # ad-hoc array
    none_mic = torch.zeros(1).type(x.type())  # fixed-array
    y1 = model(x, num_mic.long())
    y2 = model(x, none_mic.long())
    print(y1.shape, y2.shape)  # (batch, nspk, length)


if __name__ == "__main__":
    model_origin = FaSNet_origin(enc_dim=64, feature_dim=64, hidden_dim=128, layer=6, segment_size=50, 
                                 nspk=2, win_len=4, context_len=16, sr=16000)

    model_TAC = FaSNet_TAC(enc_dim=64, feature_dim=64, hidden_dim=128, layer=4, segment_size=50, 
                           nspk=2, win_len=4, context_len=16, sr=16000)

    test_model(model_origin)
    test_model(model_TAC)
