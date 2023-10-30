import torch
import torch.nn as nn
from utils import torch, device

#TODO: CharLSTM refactor
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_layers=2, dropout=0.1, conditional=1):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.num_layers = num_layers
        self.conditional = conditional

    def forward(self, input_indices, hcn=None):
        input_embedded = self.embedding(input_indices * self.conditional)
        output, (hn, cn) = self.rnn(input_embedded, hcn)        
        output = self.dropout(output)
        output = self.fc(output)
        return output

    def generate_batch(self, hcn, vocab, start='[', steps=13):
        input_indices = vocab.to_input_tensor(start).repeat(hcn[0].shape[1], 1)
        all_outputs = []
        for _ in range(steps):
            input_embeddings = self.embedding(input_indices * self.conditional)
            output, hcn = self.rnn(input_embeddings, hcn)
            output = self.dropout(output)
            output = self.fc(output)
            probabilities = torch.softmax(output, dim=-1)
            output_indices = probabilities.argmax(-1)
            all_outputs.append(output_indices)
            input_indices = output_indices
        return torch.cat(all_outputs, -1)

    def generate_beams(self, hcn, vocab, start='[', steps=13, beams=16, temperature=0.1):        
        input_indices = vocab.to_input_tensor(start).repeat(beams, 1)
        hn, cn = hcn
        hn, cn = hn[:,0:1,:], cn[:,0:1,:]
        hn, cn = hn.repeat(1, beams, 1), cn.repeat(1, beams, 1)
        hcn = (hn, cn)
        #print("hcn", hn.shape, cn.shape, input_indices.shape)
        all_outputs = []
        psteps = []
        for _ in range(steps):
            input_embeddings = self.embedding(input_indices)
            output, hcn = self.rnn(input_embeddings, hcn)
            output = self.dropout(output)
            output = self.fc(output)
            output = output/temperature
            probabilities = torch.softmax(output, dim=-1)
            pstep = probabilities
            psteps.append(pstep)
            output_indices = probabilities.argmax(-1)
            #print(output_indices.shape, probabilities.shape)
            output_indices = torch.multinomial(probabilities.squeeze(), num_samples=1)
            #print(output_indices.shape)            
            all_outputs.append(output_indices)
            input_indices = output_indices
        psteps = torch.cat(psteps, dim=1)
        #print("psteps", psteps.shape)
        #print((psteps[:,0].clamp(min=0.001)*1000).long())
        psteps_max, _ = psteps.max(-1)
        psteps_max = psteps_max[:,:-1].reshape(-1, 4, 3)
        #print(psteps_max)
        ret = torch.cat(all_outputs, -1)
        #print(ret)
        return ret
        

class DRMNet(nn.Module):
    def __init__(self):
        super(DRMNet, self).__init__()        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 18, 128)  # 32 filters, image size is reduced by half after two max pooling
        self.fc2 = nn.Linear(128, 128)  # output is a 3-dimensional vector (Euler angles)
        self.fc22 = nn.Linear(128, 128)  # output is a 3-dimensional vector (Euler angles)

    def forward(self, x, shift=0):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv12(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv22(x))
        x = nn.functional.max_pool2d(x, 2)        
        x = x.view(-1, self.num_flat_features(x))  # flatten the tensor
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc22(x)   
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

import torch
import torch.nn as nn
import torchvision.models as models

class DRMResNet(nn.Module):
    def __init__(self):
        super(DRMResNet, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.upsample = nn.Upsample((224,224), mode='nearest')
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 128)

    def forward(self, x, shift=0):
        x = self.upsample(x)
        return self.resnet18(x)        

import torch
import torch.nn as nn
import torchvision.models as models

class MyResNet(nn.Module):
    def __init__(self, num_classes=128):
        super(MyResNet, self).__init__()

        # First convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Define the basic blocks
        self.block1 = models.resnet.BasicBlock(64, 64)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        #self.fc = nn.Linear(64 * 72 * 6, num_classes)  
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Pass through the blocks
        x = self.block1(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class MyResNet1(nn.Module):
    def __init__(self, num_classes=128):
        super(MyResNet1, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = models.resnet.BasicBlock(64, 64)
        self.fc = nn.Linear(64 * 72 * 6, num_classes)  
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.block1(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class MyResNet2(nn.Module):
    def __init__(self, num_classes=128):
        super(MyResNet2, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = models.resnet.BasicBlock(64, 64)
        self.block2 = models.resnet.BasicBlock(64, 64)
        self.fc = nn.Linear(64 * 72 * 6, num_classes)  
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class MyResNetDropout(nn.Module):
    def __init__(self, num_classes=128, dropout=0.1):
        super(MyResNetDropout, self).__init__()

        # First convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Define the basic blocks
        self.block1 = models.resnet.BasicBlock(64, 64)

        # Classifier
        self.fc = nn.Linear(64 * 72 * 6, num_classes)  # Modify this line to match your actual output dimensions after the conv blocks

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Pass through the blocks with dropout
        x = self.block1(x)
        x = self.dropout(x)

        # Flatten the tensor and pass through the fully connected layer
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dropout(x)

        return x


class DRMLSTM(nn.Module):
    def __init__(self, num_chars, num_layers=4, dropout=0.1, conditional=1):
        super(DRMLSTM, self).__init__()        
        #self.model = MyResNetDropout(dropout=dropout)
        self.model = DRMNet()
        self.lstm = CharLSTM(num_chars, num_layers=num_layers, dropout=dropout, conditional=conditional).to(device)
        self.embeddings = nn.Embedding(72, 128)

    def forward(self, X, Ysrc, shift):
      embs = self.embeddings(shift.long())            
      h = embs + self.model(X)
      hn = h.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
      cn = h.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1) 
      return self.lstm(Ysrc, (hn,cn))

    def generate_batch(self, X, vocab, shift):
        if type(shift)==type(0):
          shift = torch.LongTensor([shift]).to(X.device)        
        embs = self.embeddings(shift.long())            
        h = self.model(X)
        h = embs + h
        hn = h.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        cn = h.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1) 
        return self.lstm.generate_batch((hn,cn), vocab);


    def generate_beams(self, X, vocab, shift, beams=16):
        if type(shift)==type(0):
          shift = torch.LongTensor([shift]).to(X.device)        
        embs = self.embeddings(shift.long())            
        h = self.model(X)
        h = embs + h
        hn = h.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        cn = h.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1) 
        return self.lstm.generate_beams((hn,cn), vocab, beams=beams);
