import torch
from utils import XWIDTH, augment_shift
from quats import Quaternions
from tqdm import trange
from torch.utils.data import DataLoader, Dataset

class DRMDataset(Dataset):
    def __init__(self, X, Y, vocab, xwidth=None):
        super(DRMDataset, self).__init__()
        self.images, self.labels = X, Y
        self.vocab = vocab
        self.XX = torch.cat([X, X], -1)
        self.xwidth = xwidth or XWIDTH
        #TODO: save and load with name

        Ys, shifts, encoded = [], [], []
        for i in trange(self.xwidth):
          _, Yi = augment_shift(X, Y, i)
          shifts.extend([i]*len(Y))
          Ys.append(Yi)
          quats = Quaternions.from_eulersS0(Yi)
          encs = Quaternions.to_strs_fast(quats, vocab)
          encoded.append(encs)
        self.Ys = torch.cat(Ys)
        self.shifts = torch.Tensor(shifts).long()
        self.encoded = torch.cat(encoded)        
        
    def __len__(self):
        return self.xwidth*len(self.images)

    def __getitem__(self, idx, shift=0):
        shift = idx//len(self.images)
        i = idx%len(self.images)
        XXimage = self.XX[i, :, :, XWIDTH-shift:2*XWIDTH-shift].float()        
        return XXimage, self.encoded[idx][:-1], self.encoded[idx][1:], self.shifts[idx]
    
    def to_loader(self, batch_size, shuffle=True, num_workers=4):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)



from datafold import DataFold
from vocab import Vocab

def test1():
    df = DataFold()
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = df.load_fold_val()
    print("Xtrain", "Ytrain", "Xval", "Yval", "Xtest", "Ytest")
    print(Xtrain.shape, Ytrain.shape, Xval.shape, Yval.shape, Xtest.shape, Ytest.shape)
    print("min", "max", "mean", "std")
    print(Xtrain.min(), Xtrain.max(), Xtrain.mean(), Xtrain.std())

    vocab = Vocab()
    print(vocab)

    ds = DRMDataset(Xtrain, Ytrain, vocab, xwidth=None)

    for X, Ysrc, Ytgt, shift in ds.to_loader(32):
        print("X", X.shape, "Ysrc", Ysrc.shape, "Ytgt", Ytgt.shape, "shift", shift.shape)
        print(shift)
        return


if __name__ == "__main__":
    test1()