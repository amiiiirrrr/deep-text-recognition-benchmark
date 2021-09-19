import torch.nn as nn
import torch
import time

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        # self.rnn = torch.nn.quantized.dynamic.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        # self.linear = torch.nn.quantized.dynamic.Linear(hidden_size * 2, output_size, bias_=True, dtype=torch.qint8)
        self.quantttt = torch.quantization.QuantStub()
        self.dequantttt = torch.quantization.DeQuantStub()

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        # sss = time.time()
        input = self.dequantttt(input)
        # print("#############",time.time() - sss)
        # print("######### input", input)
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        recurrent = self.quantttt(recurrent)
        # print("######### recurrent", recurrent)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output
