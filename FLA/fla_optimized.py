import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections.abc import Iterable

class FLAttention(nn.Module):
    '''
    A faithful attempt at feature level attention.
    One (Q,K,V) triplet per head.
    We don't need key betas:
    |aq*xi+bq - ak*xj+bk| = |aq*xi-ak*xj+(bq+bk)|
    Absolute value of bq determines the effect of the softmax. 
    The larger bq, the less the softmax differentiates entries in sim matrix.
    Experiment with projecting each feature into the same semantic space before FLA.
    Once for every head?
    '''
    def __init__(self, dim, num_heads, agg='sum', bounds=0.1, same_sem=True):
        super(FLAttention, self).__init__()
        self.input_dim = dim
        self.num_heads = num_heads
        self.agg = agg
        self.bounds = bounds
        self.same_sem = same_sem # refers to putting features in the same semantic space before attention

        self.alphas = nn.ParameterDict()
        self.betas = nn.ParameterDict()

        self.alphas['query'] = nn.Parameter(torch.Tensor(1,self.num_heads))
        self.alphas['key'] = nn.Parameter(torch.Tensor(1,self.num_heads))
        self.alphas['value'] = nn.Parameter(torch.Tensor(self.num_heads,))

        self.betas['query'] = nn.Parameter(torch.Tensor(1,self.num_heads))
        self.betas['value'] = nn.Parameter(torch.Tensor(1, num_heads))

        self.ones = torch.ones(dim,1)
        if self.same_sem:
            self.sem_weights = nn.Parameter(torch.Tensor(self.input_dim))
            self.sem_biases = nn.Parameter(torch.Tensor(self.input_dim))
            
        if self.agg =='proj':
            self.proj = nn.Linear(self.input_dim*self.num_heads, self.input_dim)

        self.reset_parameters()

    def reset_parameters(self):
        #Learn more about initializations.
        for key in self.alphas.keys():
            nn.init.uniform_(self.alphas[key], a=1-self.bounds, b=1+self.bounds)
        for key in self.betas.keys():
            nn.init.uniform_(self.betas[key], a=-self.bounds, b=self.bounds)
    
        if self.same_sem:
            nn.init.uniform_(self.sem_weights, a=1-self.bounds, b=1+self.bounds)
            nn.init.uniform_(self.sem_biases, a=-self.bounds, b=self.bounds)       
        
        
    def compute_sim(self, x, y, epsilon=1e-8):
        x = x.unsqueeze(1)
        y = y.unsqueeze(2)
        diff_matrix = torch.abs(x - y)
        diff_matrix += epsilon
        sim_matrix = torch.reciprocal(diff_matrix)
        sim_matrix = F.softmax(sim_matrix, dim=-2) / torch.sqrt(torch.tensor(sim_matrix.shape[-1]))
        # sum_of_rows = diff_matrix.sum(dim=-2, keepdim=True)
        # sim_matrix = (1-diff_matrix/sum_of_rows) / torch.sqrt(torch.tensor(diff_matrix.shape[-1]))
        # sim_matrix = (F.softmin(diff_matrix, dim=-2))/ torch.sqrt(torch.tensor(diff_matrix.shape[-1]))
        #doesn't blow up values near 0
        # sim_matrix = (1-F.softmax(diff_matrix, dim=-2))/ torch.sqrt(torch.tensor(diff_matrix.shape[-1]))
        return sim_matrix
    
    def forward(self, x):
        if self.num_heads == 0:
            return x
        else:
            if self.same_sem:
                x = x * self.sem_weights + self.sem_biases
            
            q = x.unsqueeze(-1) @ self.alphas['query'] + self.ones @ self.betas['query']
            k = x.unsqueeze(-1) @ self.alphas['key']
            similarity_matrix = self.compute_sim(k, q)
        
            if self.agg=='sum':
                attended_value = torch.sum(similarity_matrix*self.alphas['value'], dim=-1)
                combined_reps = torch.bmm(attended_value, x.unsqueeze(-1)).squeeze(-1)
                combined_reps += torch.sum(torch.sum(similarity_matrix,dim=-2)*self.betas['value'],dim=-1)
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
            self.flas = nn.ModuleList([FLAttention(dims[i], num_heads=self.attn_heads, agg=self.agg) for i in range(len(dims))
                                        ])
            self.fla_norms = nn.ModuleList([nn.LayerNorm(dims[i]) for i in range(len(dims))
                                        ])
        
        self.activation = activation()

        self.output = nn.Linear(dims[-1], output_dim)
    
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

class FLALR(nn.Module):
    '''
    FLA + Logistic Regression. Could augment FLANN to do this but whatever.
    '''
    def __init__(self, input_dim, num_heads, agg='sum', activation=nn.ReLU):
        super(FLALR, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.agg=agg

        self.fla = FLAttention(dim=self.input_dim, num_heads=self.num_heads, agg=self.agg)
        self.lr = nn.Linear(input_dim, 1)
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(self.input_dim)
    def forward(self,x):
        if self.num_heads != 0:
            x = self.layer_norm(self.activation(self.fla(x)))
        return self.lr(x)

class UFLAttention(nn.Module):
    '''
    A less faithful attempt at feature level attention.
    "U" refers to the Unique affine transformations of each feature.
    For every head, there is a unique query transformation for every feature
    and unique key and value transformations for every pair of features.
    '''
    def __init__(self, dim, num_heads, aggregation='sum', activation=nn.ReLU):
        super(UFLAttention, self).__init__()
        self.dim = dim 
        self.num_heads = num_heads
        self.aggregation = aggregation

        self.query_weights = nn.Parameter(torch.ones(dim, num_heads))
        self.key_weights = nn.Parameter(torch.ones(dim,dim,num_heads))
        #try different things here
        # self.value_weights = nn.Parameter(torch.ones(1,num_heads)) #uniform transformation for features
        self.value_weights = nn.Parameter(torch.ones(dim, num_heads))
        
        self.query_biases = nn.Parameter(torch.zeros(dim, num_heads))
        self.key_biases = nn.Parameter(torch.zeros(dim,dim,num_heads))
        self.value_biases = nn.Parameter(torch.zeros(dim, num_heads))
        # self.value_biases = nn.Parameter(torch.zeros(1,num_heads)) #uniform transformation for features
        # self.ones = torch.ones(dim, 1)

        if aggregation == 'prod':
            self.proj = nn.Linear(dim*num_heads, num_heads)
        
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(dim)
        
    def compute_sim(self, q, k, epsilon=1e-8):

        diff_matrix = torch.abs(q.unsqueeze(2)-k)
        diff_matrix += epsilon
        sim_matrix = 1.0 / diff_matrix

        sim_matrix = F.softmax(sim_matrix, dim=-2) / torch.sqrt(torch.tensor(k.shape[1]))
        return sim_matrix
    
    def forward(self, x):
        if self.num_heads == 0:
            return x
        else:
            diag_x = torch.stack([torch.diag(x[i]) for i in range(x.shape[0])])
            q = torch.matmul(diag_x, self.query_weights) + self.query_biases
            key_weights_expanded = self.key_weights.unsqueeze(0).expand(x.size(0), -1, -1, -1)
            k = torch.einsum('bij,bjkh -> bikh', diag_x, key_weights_expanded) + self.key_biases

            similarity_matrix = self.compute_sim(q,k)
            v =  torch.matmul(diag_x, self.value_weights) + self.value_biases
            attended_value = torch.matmul(similarity_matrix, v.unsqueeze(-1)).squeeze(-1)
            # v = x.unsqueeze(-1) @ self.value_weights + self.ones @ self.value_biases #uniform transformation for features
            # attended_value = torch.einsum('bijc,bjc->bic', similarity_matrix, v) #uniform transformation for features
            
            if self.aggregation == 'sum':
                combined_reps = torch.sum(attended_value, dim=-1)
            elif self.aggregation == 'proj':
                combined_reps = torch.cat([attended_value[:,:,i] for i in range(attended_value.size(-1))],axis=-1)
                combined_reps = self.proj(combined_reps)
            
            contextual_reps = x + combined_reps
            contextual_reps = self.activation(contextual_reps)
            contextual_reps = self.layer_norm(contextual_reps)
            return contextual_reps


class UFLANN(nn.Module):
    def __init__(self, input_dim, hidden_dims, attn_heads, activation, agg='sum', output_dim=1):
        super(UFLANN, self).__init__()
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
            self.uflas = nn.ModuleList([UFLAttention(dims[i], num_heads=self.attn_heads, aggregation=self.agg) for i in range(len(dims)-1)
                                        ])
        
        self.activation = activation()

        self.output = nn.Linear(dims[-1], output_dim)
    
    def forward(self, x):
        for i in range(len(self.linears)):
            if self.attn_heads!=0:
                x = self.uflas[i](x)
            x = self.linears[i](x)
            x = self.activation(x)
            x = self.linear_norms[i](x)
        x = self.output(x) # torch crossentropy automatically activates. BCEWithLogitsLoss for binary.
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
    '''
    Maybe add a softmax.
    '''
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
    def __init__(self, input_dim, n_heads, agg='proj', activation=nn.ReLU):
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

            #multiple heads reduces to one head
            if self.agg=='sum':
                x = x + torch.stack([alphas[i]*x+betas[i] for i in range(self.n_heads)],dim=-1).sum(dim=-1)
            elif self.agg=='proj':
                if self.n_heads > 1:
                    x = x + self.proj(torch.cat([alphas[i]*x+betas[i] for i in range(self.n_heads)], dim=-1))
                else:
                    x = x + alphas[0]*x + betas[0]
            
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

class LinearSoftmax(nn.Module):
    '''
    Not FLA. Experimenting with applying operations to a weight matrix of a linear layer before
    multiplying by a feature vector.
    '''
    def __init__(self,input_dim, output_dim, activation):
        super(LinearSoftmax, self).__init__()
        
        self.weight = nn.Parameter(torch.rand(output_dim, input_dim))
        # self.weight /= torch.sum(self.weight, dim=0)
        self.bias = nn.Parameter(torch.zeros(output_dim))

        self.softmax = nn.Softmax(dim=1)
        self.activation = activation()

    def forward(self, x):
        weights_prob = self.softmax(self.weight)

        out = F.linear(x, weights_prob, self.bias)
        return self.activation(out)
    
class GaussianActivation(nn.Module):
    def __init__(self, mean=0, std=1):
        super(GaussianActivation, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return torch.exp((-(x - self.mean) ** 2)/(2* self.std ** 2))

class MultiActivationLinear(nn.Module):
    '''
    Not FLA. Experimenting with making linear layers by concatenating mini-layers with different
    activations.
    '''
    def __init__(self, activations, input_dim, nodes_per_activation):
        super(MultiActivationLinear, self).__init__()
        self.input_dim  = input_dim
        self.nodes_per_activation = nodes_per_activation
        if isinstance(self.nodes_per_activation, Iterable):
            assert len(nodes_per_activation) == len(activations)
            self.linears = nn.ModuleList([nn.Linear(in_features=input_dim, out_features=nodes) for nodes in self.nodes_per_activation])
            self.batchnorms = nn.ModuleList([nn.BatchNorm1d(node) for node in nodes_per_activation])
        else:
            self.linears = nn.ModuleList([nn.Linear(in_features=input_dim, out_features=self.nodes_per_activation) for x in activations])
            self.batchnorms = nn.ModuleList([nn.BatchNorm1d(nodes_per_activation) for x in activations])
        self.activations = nn.ModuleList([activation() for activation in activations])
        
    def forward(self, x):
        x = torch.concat([a(n(l(x))) for (a,n,l) in zip(self.activations, self.batchnorms, self.linears)], dim=1)
        return x