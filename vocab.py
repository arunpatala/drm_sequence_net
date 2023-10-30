import torch
from utils import device

class Vocab:

  def __init__(self, chars="0123456789[]-,.", sort=True):
    
    self.chars = list(chars)
    if sort: self.chars = list(sorted(set(self.chars)))
    self.char2int = {ch: i for i, ch in enumerate(self.chars)}
    self.int2char = {i: ch for i, ch in enumerate(self.chars)}
    self.str = "".join(self.chars)
    self.len = len(self.chars)

  def __len__(self):
    return self.len

  def __repr__(self):
    return self.str

  def to_list(self, data):
    return [self.char2int[s] for s in data]

  def to_input_tensor(self, input_str):
    input_tensor = [self.char2int[s] for s in input_str]
    input_tensor = torch.LongTensor(input_tensor).unsqueeze(0)
    return input_tensor.to(device)

  def tensor_to_str(self, tensor):
      output_str = ""
      for index in tensor.squeeze():
          char = self.int2char[index.item()]
          output_str += char
      return output_str
  
  def find(self, char):
     return self.str.find(char)
  
  def encode(self, t):
        vocab = self    
        t = t + vocab.find("0")
        N = len(t)
        begin = vocab.find('[')
        begin_tensor = torch.full((N, 1), begin, dtype=torch.long).to(t.device)
        end = vocab.find(']')
        end_tensor = torch.full((N, 1), end, dtype=torch.long).to(t.device)
        t = torch.cat([begin_tensor, t, end_tensor], dim=1)
        return t


def test1():
   vocab = Vocab()
   print(vocab)
   print("0", vocab.find("0"))


if __name__ == "__main__":
    test1()

