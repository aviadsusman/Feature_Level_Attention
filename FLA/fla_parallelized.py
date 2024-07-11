import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import time

class FLAttention(nn.Module):
    '''
    A faithful attempt at feature level attention.
    One (Q,K,V) triplet per head.
    '''
    def __init__(self, dim, num_heads, agg='sum'):
        super(FLAttention, self).__init__()
        self.input_dim = dim
        self.num_heads = num_heads
        self.agg = agg

        self.alphas = nn.ParameterDict()
        self.betas = nn.ParameterDict()
        self.transformations = ['query', 'key', 'value']
        for transform in self.transformations:
            self.alphas[transform] = nn.Parameter(torch.Tensor(1,self.num_heads))
            self.betas[transform] = nn.Parameter(torch.Tensor(1,self.num_heads))
        
        self.ones = torch.ones(dim, 1)

        if self.agg =='proj':
            self.proj = nn.Linear(self.input_dim*self.num_heads, self.input_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        #Learn more about initializations.
        for transform in self.transformations:
            nn.init.uniform_(self.alphas[transform])
            #zeros or uniform?
            nn.init.zeros_(self.betas[transform])
        
    def compute_sim(self, x, y, epsilon=1e-8):
        x = x.unsqueeze(1)
        y = y.unsqueeze(2)
        diff_matrix = torch.abs(x - y)
        diff_matrix += epsilon
        sim_matrix = torch.reciprocal(diff_matrix)
        sim_matrix = F.softmax(sim_matrix, dim=-2) / torch.sqrt(torch.tensor(sim_matrix.shape[-1]))
        return sim_matrix
    
    def forward(self, x):
        if self.num_heads == 0:
            return x
        else:
            q = x.unsqueeze(-1) @ self.alphas['query'] + self.ones @ self.betas['query']
            k = x.unsqueeze(-1) @ self.alphas['key'] + self.ones @ self.betas['key']
            v = x.unsqueeze(-1) @ self.alphas['value'] + self.ones @ self.betas['value']
            similarity_matrix = self.compute_sim(k, q)
            attended_value = torch.einsum('bijc,bjc->bic', similarity_matrix, v)
        
        if self.agg=='sum':
            combined_reps = torch.sum(attended_value, dim=-1)
        elif self.agg=='proj':
            combined_reps = torch.cat([attended_value[:,:,i] for i in range(attended_value.size(-1))],axis=-1)
            combined_reps = self.proj(combined_reps)
        
        return x + combined_reps
        
class FLANN(nn.Module):
    def __init__(self, input_dim, hidden_dims, attn_heads, activation, agg='sum', output_dim=1):
        super(FLANN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.attn_heads = attn_heads
        self.agg=agg

        dims = [input_dim] + list(hidden_dims)
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
                                      ])
        self.linear_norms = nn.ModuleList([nn.LayerNorm(dims[i+1]) for i in range(len(dims)-1)
                                      ])
        
        if self.attn_heads!=0:
            self.flas = nn.ModuleList([FLAttention(dims[i], num_heads=self.attn_heads, agg=self.agg) for i in range(len(dims)-1)
                                        ])
            self.fla_norms = nn.ModuleList([nn.LayerNorm(dims[i]) for i in range(len(dims)-1)
                                        ])
        
        self.activation = activation()

        self.output = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, x):
        for i in range(len(self.linears)):
            if self.attn_heads!=0:
                x = self.flas[i](x)
                x = self.activation(x)
                x = self.fla_norms[i](x)
            x = self.linears[i](x)
            x = self.activation(x)
            x = self.linear_norms[i](x)
        x = self.output(x) # torch crossentropy automatically activates. BCEWithLogitsLoss for binary.
        return x

class UFLAttention(nn.Module):
    '''
    A less faithful attempt at feature level attention.
    "Unique" refers to the unique affine transformations of every feature into (Q,K,V) representations.
    '''
    def __init__(self, dim, num_heads):
        super(UFLAttention, self).__init__()

        self.num_heads = num_heads

        self.weights = nn.ParameterDict()
        self.biases = nn.ParameterDict()
        self.transformations = ['query', 'key', 'value']
        for head in range(self.num_heads):
            for transform in self.transformations:
                self.weights[f'{transform}_{head}'] = nn.Parameter(torch.Tensor(dim))
                self.biases[f'{transform}_{head}'] = nn.Parameter(torch.Tensor(dim))
        self.reset_parameters()

    def reset_parameters(self):
        for head in range(self.num_heads):
            for transform in self.transformations:
                nn.init.uniform_(self.weights[f'{transform}_{head}'])
                nn.init.zeros_(self.biases[f'{transform}_{head}'])
        
    def compute_sim(self, x, y, epsilon=1e-8):
        x = x.unsqueeze(1)
        y = y.unsqueeze(2)

        diff_matrix = torch.abs(x - y)
        diff_matrix += epsilon
        sim_matrix = 1.0 / diff_matrix

        sim_matrix = F.softmax(sim_matrix, dim=-1) / torch.sqrt(torch.tensor(y.shape[1]))
        return sim_matrix
    
    def forward(self, x):
        if self.num_heads == 0:
            return x
        else:
            representations = []
            for head in range(self.num_heads):
                head_representations = {}
                for t in self.transformations:
                    head_representations[t] = self.weights[f'{t}_{head}'] * x + self.biases[f'{t}_{head}']
                
                query = head_representations['query']
                key = head_representations['key']
                value = head_representations['value']

                similarity_matrix = self.compute_sim(query, key)
                
                attended_value = torch.bmm(similarity_matrix, value.unsqueeze(-1)).squeeze(-1)
                representations.append(attended_value)
            
            combined_representations = torch.stack(representations, dim=0).sum(dim=0)
            return x + combined_representations
    

class UFLANN(nn.Module):
    def __init__(self, input_dim, attn_heads, hidden_dims, activation, output_dim=1):
        super(UFLANN, self).__init__()

        self.utab_attn = UFLAttention(dim=input_dim, num_heads=attn_heads)
        self.attn_norm = nn.LayerNorm(input_dim)

        dims = [input_dim] + list(hidden_dims)
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
                                      ])
        self.linear_norms = nn.ModuleList([nn.LayerNorm(dims[i+1]) for i in range(len(dims)-1)
                                      ])
        self.activation = activation

        self.output = nn.Linear(hidden_dims[-1], output_dim)

    
    def forward(self, x):
        x = self.utab_attn(x)
        x = self.activation(x)
        x = self.attn_norm(x)
        for i, layer in enumerate(self.linears):
            x = layer(x)
            x = self.activation(x)
            x = self.linear_norms[i](x)
        x = self.output(x)
        return x

class MacroF1Score(nn.Module):
    def __init__(self, num_classes):
        super(MacroF1Score, self).__init__()
        self.num_classes = num_classes
        self.epsilon = 1e-7

    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        y_true = y_true.float()
        
        f1_scores = []
        for i in range(self.num_classes):
            TP = ((y_pred.argmax(dim=1) == i) & (y_true[:, i] == 1)).sum().float()
            FP = ((y_pred.argmax(dim=1) == i) & (y_true[:, i] == 0)).sum().float()
            FN = ((y_pred.argmax(dim=1) != i) & (y_true[:, i] == 1)).sum().float()

            precision = TP / (TP + FP + self.epsilon)
            recall = TP / (TP + FN + self.epsilon)

            f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
            f1_scores.append(f1)

        macro_f1 = torch.mean(torch.stack(f1_scores))

        return macro_f1


class npDataset(Dataset):
    '''
    For converting numpy arrays to torch datasets
    '''
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

class HFLANN(nn.Module):
    '''
    An implementation of FLA in which Q and K projections of each feature have length > 1.
    Can make use of standard ScaledDotProduct attention by reshaping data from (batch, features)
    to (batch, features, 1).
    '''
    def __init__(self, input_dim, embed_dim, hidden_dims, attn_heads, activation, output_dim=1):
        super(HFLANN, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.attn_heads = attn_heads
        self.activation = activation
        


        dims = [input_dim] + list(hidden_dims)
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
                                      ])
        self.linear_norms = nn.ModuleList([nn.LayerNorm(dims[i+1]) for i in range(len(dims)-1)
                                      ])
        
        if self.attn_heads!=0:
            # self.flas = nn.ModuleList([nn.MultiheadAttention(
            #     dims[i], num_heads=self.attn_heads) for i in range(len(dims)-1
            #                                                        )])
            self.flas = nn.ModuleList([FLAttention(dims[i], num_heads=self.attn_heads) for i in range(len(dims)-1)
                                        ])
            self.fla_norms = nn.ModuleList([nn.LayerNorm(dims[i]) for i in range(len(dims)-1)
                                        ])
        
        self.activation = activation

        self.output = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, x):
        for i in range(len(self.linears)):
            if self.attn_heads!=0:
                x = self.flas[i](x)
                x = self.activation(x)
                x = self.fla_norms[i](x)
            x = self.linears[i](x)
            x = self.activation(x)
            x = self.linear_norms[i](x)
        x = self.output(x) # torch crossentropy automatically activates. BCEWithLogitsLoss for binary.
        return x


class MultiLinear(nn.Module):
    def __init__(self, n_heads, input_dim, agg='sum'):
        super(MultiLinear, self).__init__()
        self.agg = agg
        self.sub_linears = nn.ModuleList([nn.Linear(input_dim,input_dim) for i in range(n_heads)])
        if agg=='proj':
            self.proj = nn.Linear(input_dim*n_heads, input_dim)
    
    def forward(self,x):
        output = x
        if self.agg=='sum':
            output = x + torch.stack([l(x) for l in self.sub_linears]).sum(dim=0)
        elif self.agg=='proj':
            output = x + self.proj(torch.cat([l(x) for l in self.sub_linears], dim=-1))
        return output

class ResNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, res_heads, activation, agg='sum'):
        super(ResNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.res_heads = res_heads
        self.agg = agg
    
        dims = [input_dim] + list(hidden_dims)
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
                                      ])
        self.linear_norms = nn.ModuleList([nn.LayerNorm(dims[i+1]) for i in range(len(dims)-1)
                                      ])
        
        if self.res_heads!=0:
            self.res_linears = nn.ModuleList([MultiLinear(input_dim=dims[i], n_heads=self.res_heads, agg=self.agg) for i in range(len(dims)-1)
                                        ])
            self.res_norms = nn.ModuleList([nn.LayerNorm(dims[i]) for i in range(len(dims)-1)
                                        ])
        
        self.activation = activation
        self.output = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        for i in range(len(self.linears)):
            if self.res_heads!=0:
                x = self.res_linears[i](x)
                x = self.activation(x)
                x = self.res_norms[i](x)
            x = self.linears[i](x)
            x = self.activation(x)
            x = self.linear_norms[i](x)
        x = self.output(x) # torch crossentropy automatically activates. BCEWithLogitsLoss for binary.
        return x
    

class SelfConditioning(nn.Module):
    '''
    Giving features contextual awareness of each by conditioning feature vector with itself.
    '''
    def __init__(self, input_dim, n_heads, agg='sum', activation=nn.ReLU):
        super(SelfConditioning, self).__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.agg = agg

        self.FiLMs = nn.ModuleList([nn.Linear(input_dim, 2*input_dim) for i in range(n_heads)])
   
        self.layer_norm = nn.LayerNorm(input_dim)

        self.activation = activation
        
        if agg == 'proj':
            self.proj = nn.Linear(n_heads*input_dim, input_dim)
    
    def forward(self, x):
        if self.n_heads == 0:
            return x
        else:
            alphas = [self.FiLMs[i](x)[:,:self.input_dim] for i in range(self.n_heads)]
            betas = [self.FiLMs[i](x)[:,self.input_dim:] for i in range(self.n_heads)]

            if self.agg=='sum':
                x = x + torch.stack([alphas[i]*x+betas[i] for i in range(self.n_heads)],dim=-1).sum(dim=-1)
            elif self.agg=='proj':
                x = x + self.proj(torch.cat([alphas[i]*x+betas[i] for i in range(self.n_heads)]))
            
            x = self.activation(x)
            return self.layer_norm(x)

class TabularAwarenessNetwork(nn.Module):
    '''
    Class for applying different contextual awareness mechanisms to each feed forward layer
    of a fully connected network.
    '''
    def __init__(self, input_dim, hidden_dims, output_dim, n_heads, agg, activation, mech='fla'):
        super(TabularAwarenessNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.outout_dim = output_dim
        self.n_heads = n_heads
        self.agg = agg
        self.mech = mech
        self.activation = activation

        dims = [input_dim] + list(hidden_dims)
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
                                      ])
        self.linear_norms = nn.ModuleList([nn.LayerNorm(dims[i+1]) for i in range(len(dims)-1)
                                      ])
        mechanisms = {'fla' : FLAttention,
                      'ufla': UFLAttention,
                      'res' : ResNN,
                      'film' : SelfConditioning
                      # 'hfla': HFLAttention
                     }
        awareness = mechanisms[self.mech]
        if self.n_heads!=0:
            self.awares = nn.ModuleList([awareness(input_dim = dims[i], n_heads=self.n_heads, agg=self.agg, activation=self.activation) for i in range(len(dims)-1)
                                        ])
            
            #build layer norms into awareness layer
            # self.fla_norms = nn.ModuleList([nn.LayerNorm(dims[i]) for i in range(len(dims)-1)
            #                             ])
        
        self.activation = activation
        self.output = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, x):
        for i in range(len(self.linears)):
            if self.n_heads!=0:
                x = self.awares[i](x)
            x = self.linears[i](x)
            x = self.activation(x)
            x = self.linear_norms[i](x)
        x = self.output(x) # torch crossentropy automatically activates. BCEWithLogitsLoss for binary.
        return x
