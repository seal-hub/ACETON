import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class AttentionModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, max_seq_len, output_dim, num_layers):
        super(AttentionModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers)
        self.label = nn.Linear(hidden_dim * max_seq_len, output_dim)
        self.sc_first = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.last = nn.Linear(2, max_seq_len, bias=False)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def attention_net(self, lstm_output, final_hidden_state):  # (b,s,h) - (1,b,h)
        hidden = final_hidden_state.squeeze(0)  # (b,h)
        sc_first = self.sc_first(lstm_output)  # (b,s,h) dot W(h,h) -> (b,s,h)
        attn_weights = torch.bmm(sc_first, hidden.unsqueeze(2)).squeeze(2)  # (b,s,h) bmm (b,h,1) -> (b,s)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context_vector = torch.bmm(lstm_output.permute(0,2,1), soft_attn_weights.unsqueeze(2))  # (b, h,s) dot (b, s,1) => (b, h,1)
        pre = torch.cat((hidden.unsqueeze(2), context_vector), dim=-1)
        attention_vector = self.last(pre) #(1,h,max_s)

        return torch.tanh(attention_vector), attn_weights

    def forward(self, inputs):
        h_0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))  # Initial hidden state of the LSTM
        c_0 = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))  # Initial cell state of the LSTM
        inputs = inputs.permute(1, 0, 2)
        lstm_out, _ = self.lstm(inputs.float(), (h_0, c_0))
        sigmoided_out = torch.relu(lstm_out)
        final_hidden_state = sigmoided_out[-1].unsqueeze(0)
        output = sigmoided_out.permute(1, 0, 2)
        attn_output, attention_weights = self.attention_net(output, final_hidden_state)
        t = attn_output.view(attn_output.size(0), -1)
        y_pred = self.label(t)
        return y_pred, F.softmax(attention_weights, 1)
