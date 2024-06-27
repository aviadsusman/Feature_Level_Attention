import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class FLAttention(nn.Module):
    '''
    A faithful attempt at feature level attention.
    One (Q,K,V) triplet per head.
    '''
    def __init__(self, input_dim, num_heads):
        super(FLAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads


        self.alphas = nn.ParameterDict()
        self.betas = nn.ParameterDict()
        self.transformations = ['query', 'key', 'value']
        for head in range(self.num_heads):
            for transform in self.transformations:
                self.alphas[f'{transform}_{head}'] = nn.Parameter(torch.Tensor(1))
                self.betas[f'{transform}_{head}'] = nn.Parameter(torch.Tensor(1))
        self.ones = torch.ones(input_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        for head in range(self.num_heads):
            for transform in self.transformations:
                nn.init.uniform_(self.alphas[f'{transform}_{head}'])
                nn.init.uniform_(self.betas[f'{transform}_{head}'])
        
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
                    head_representations[t] = self.alphas[f'{t}_{head}'] * x + self.betas[f'{t}_{head}'] * self.ones
                
                query = head_representations['query']
                key = head_representations['key']
                value = head_representations['value']


                similarity_matrix = self.compute_sim(query, key)
                
                attended_value = torch.bmm(similarity_matrix, value.unsqueeze(-1)).squeeze(-1)
                representations.append(attended_value)
            
            combined_representations = torch.stack(representations, dim=0).sum(dim=0)
            return x + combined_representations
        
class FLANN(nn.Module):
    def __init__(self, input_dim, hidden_dims, attn_heads, activation, output_dim=1):
        super(FLANN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.attn_heads = attn_heads
        self.activation = activation

        dims = [input_dim] + list(hidden_dims)
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
                                      ])
        self.linear_norms = nn.ModuleList([nn.LayerNorm(dims[i+1]) for i in range(len(dims)-1)
                                      ])
        
        self.flas = nn.ModuleList([FLAttention(dims[i], num_heads=self.attn_heads) for i in range(len(dims)-1)
                                      ])
        self.fla_norms = nn.ModuleList([nn.LayerNorm(dims[i]) for i in range(len(dims)-1)
                                      ])
        
        self.activation = activation

        self.output = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, x):
        for i in range(len(self.flas)):
            x = self.flas[i](x)
            x = self.activation(x)
            x = self.fla_norms[i](x)
            x = self.linears[i](x)
            x = self.activation(x)
            x = self.linear_norms[i](x)
        x = self.output(x) # torch crossentropy activates with softmax
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
    

class UTAMLP(nn.Module):
    def __init__(self, input_dim, attn_heads, hidden_dims, activation, output_dim=1):
        super(UTAMLP, self).__init__()

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
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    




