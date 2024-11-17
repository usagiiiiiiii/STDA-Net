import torch
import torch.nn as nn

from .stda_encoder import PretrainingEncoder

# initilize weight
def weights_init(model):
    with torch.no_grad():
        for child in list(model.children()):
            for param in list(child.parameters()):
                  if param.dim() == 2:
                        nn.init.xavier_uniform_(param)
    print('weights initialization finished!')

class STDA_Net(nn.Module):
    def __init__(self, args_encoder, dim=3072, K=65536, m=0.999, T=0.07):
        """
        args_encoder: model parameters encoder
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(STDA_Net, self).__init__()

        self.K = K
        self.m = m
        self.T = T
  
        print(" moco parameters",K,m,T)

        self.encoder_q = PretrainingEncoder(**args_encoder)
        self.encoder_k = PretrainingEncoder(**args_encoder)
        weights_init(self.encoder_q)
        weights_init(self.encoder_k)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue

        # domain level queues
        # temporal domain queue
        self.register_buffer("t_queue", torch.randn(dim, K))
        self.t_queue = nn.functional.normalize(self.t_queue, dim=0)
        self.register_buffer("t_queue_ptr", torch.zeros(1, dtype=torch.long))

        # spatial domain queue
        self.register_buffer("s_queue", torch.randn(dim, K))
        self.s_queue = nn.functional.normalize(self.s_queue, dim=0)
        self.register_buffer("s_queue_ptr", torch.zeros(1, dtype=torch.long))

        # clip level queue
        self.register_buffer("ft_queue", torch.randn(dim, K))
        self.ft_queue = nn.functional.normalize(self.ft_queue, dim=0)
        self.register_buffer("ft_queue_ptr", torch.zeros(1, dtype=torch.long))

        # part level queue
        self.register_buffer("fs_queue", torch.randn(dim, K))
        self.fs_queue = nn.functional.normalize(self.fs_queue, dim=0)
        self.register_buffer("fs_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, t_keys, s_keys, ft_keys, fs_keys):

        N, C = t_keys.shape
        assert self.K % N == 0  # for simplicity

        t_ptr = int(self.t_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.t_queue[:, t_ptr:t_ptr + N] = t_keys.T
        t_ptr = (t_ptr + N) % self.K  # move pointer
        self.t_queue_ptr[0] = t_ptr

        s_ptr = int(self.s_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.s_queue[:, s_ptr:s_ptr + N] = s_keys.T
        s_ptr = (s_ptr + N) % self.K  # move pointer
        self.s_queue_ptr[0] = s_ptr

        ft_ptr = int(self.ft_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.ft_queue[:, ft_ptr:ft_ptr + N] = ft_keys.T
        ft_ptr = (ft_ptr + N) % self.K  # move pointer
        self.ft_queue_ptr[0] = ft_ptr

        fs_ptr = int(self.fs_queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.fs_queue[:, fs_ptr:fs_ptr + N] = fs_keys.T
        fs_ptr = (fs_ptr + N) % self.K  # move pointer
        self.fs_queue_ptr[0] = fs_ptr

    def forward(self, xq, xk):
        """
        Input:
            time-majored domain input sequence: qc_input and kc_input
            space-majored domain input sequence: qp_input and kp_input
        Output:
            logits and targets
        """

        # compute clip level, part level, temporal domain level, spatial domain level and instance level features
        qt, qs, qft, qfs = self.encoder_q(xq)  # queries: NxC

        qt = nn.functional.normalize(qt, dim=1)
        qs = nn.functional.normalize(qs, dim=1)
        qft = nn.functional.normalize(qft, dim=1)
        qfs = nn.functional.normalize(qfs, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            kt, ks, kft, kfs = self.encoder_k(xk)	  # keys: NxC

            kt = nn.functional.normalize(kt, dim=1)
            ks = nn.functional.normalize(ks, dim=1)
            kft = nn.functional.normalize(kft, dim=1)
            kfs = nn.functional.normalize(kfs, dim=1)
     
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: NxL
        l_pos_t = torch.einsum('nc,nc->n', [qt, kft]).unsqueeze(1)
        l_pos_s = torch.einsum('nc,nc->n', [qs, kfs]).unsqueeze(1)
        l_pos_f1 = torch.einsum('nc,nc->n', [qft, kt]).unsqueeze(1)
        l_pos_f2 = torch.einsum('nc,nc->n', [qfs, ks]).unsqueeze(1)

        l_pos_i1 = torch.einsum('nc,nc->n', [qt, ks]).unsqueeze(1)
        l_pos_i2 = torch.einsum('nc,nc->n', [qs, kt]).unsqueeze(1)

        # negative logits: NxK
        l_neg_t = torch.einsum('nc,ck->nk', [qt, self.ft_queue.clone().detach()])
        l_neg_s = torch.einsum('nc,ck->nk', [qs, self.fs_queue.clone().detach()])
        l_neg_f1 = torch.einsum('nc,ck->nk', [qft, self.t_queue.clone().detach()])
        l_neg_f2 = torch.einsum('nc,ck->nk', [qfs, self.s_queue.clone().detach()])

        l_neg_i1 = torch.einsum('nc,ck->nk', [qt, self.s_queue.clone().detach()])
        l_neg_i2 = torch.einsum('nc,ck->nk', [qs, self.t_queue.clone().detach()])

        # logits: Nx(1+K)
        logits_t = torch.cat([l_pos_t, l_neg_t], dim=1)
        logits_s = torch.cat([l_pos_s, l_neg_s], dim=1)
        logits_f1 = torch.cat([l_pos_f1, l_neg_f1], dim=1)
        logits_f2 = torch.cat([l_pos_f2, l_neg_f2], dim=1)

        logits_i1 = torch.cat([l_pos_i1, l_neg_i1], dim=1)
        logits_i2 = torch.cat([l_pos_i2, l_neg_i2], dim=1)

        # apply temperature
        logits_t /= self.T
        logits_s /= self.T
        logits_f1 /= self.T
        logits_f2 /= self.T

        logits_i1 /= self.T
        logits_i2 /= self.T


        # positive key indicators
        labels_t = torch.zeros(logits_t.shape[0], dtype=torch.long).cuda()
        labels_s = torch.zeros(logits_s.shape[0], dtype=torch.long).cuda()
        labels_f1 = torch.zeros(logits_f1.shape[0], dtype=torch.long).cuda()
        labels_f2 = torch.zeros(logits_f2.shape[0], dtype=torch.long).cuda()

        labels_i1 = torch.zeros(logits_i1.shape[0], dtype=torch.long).cuda()
        labels_i2 = torch.zeros(logits_i2.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(kt, ks, kft, kfs)

        return logits_t, logits_s, logits_f1, logits_f2, logits_i1, logits_i2, \
                labels_t, labels_s, labels_f1, labels_f2, labels_i1, labels_i2