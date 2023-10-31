import torch
import numpy as np
from utils import normalize_images

class DataFold:
    def __init__(self, root='DATA/data', train_set='training_sets', eval_set='evaluation_sets'):
        self.root = root
        self.train_set = train_set
        self.eval_set = eval_set

    def load_data(self, range_values=range(1, 11), sets=None):
        if sets is None:
            sets = self.train_set
        X, Y = [], []
        for i in range_values:
            data = np.load(f'{self.root}/{sets}/{i:02d}.npy', allow_pickle=True).item()
            X.append(data['xtr'])
            Y.append(data['ytr'])            
        X = torch.from_numpy(np.concatenate(X, axis=0)).unsqueeze(1).float()
        Y = torch.from_numpy(np.concatenate(Y, axis=0))        
        return normalize_images(X), Y

    def load_data_splits(self, splits, sets=None):
        return self.load_data(splits, sets)

    def load_fold(self, split=1):
        splits = [i for i in range(1, 11) if i != split]
        Xt, Yt = self.load_data_splits(splits)
        Xtt, Ytt = self.load_data_splits([split], sets=self.eval_set)        
        return Xt, Yt, Xtt, Ytt

    def load_fold_val(self, split=1):
        splits = [i for i in range(1, 11) if i != split]
        Xt, Yt = self.load_data_splits(splits)
        Xv, Yv = self.load_data_splits([split])
        Xtt, Ytt = self.load_data_splits([split], sets=self.eval_set)        
        return Xt, Yt, Xv, Yv, Xtt, Ytt
    
    def load_all(self, to_tensor=True, sample='', slice=100, x=0, y=0):
        data = np.load(f'{self.root}/samples/08/drm_data{sample}.npy')        
        eulers = np.load(f'{self.root}/samples/08/eulers{sample}.npy')
        if slice is not None:
          data = data[x:x+slice,y:y+slice]
          eulers = eulers[x:x+slice,y:y+slice]
        if to_tensor:
            #return normalize_images(torch.from_numpy(data).unsqueeze(1).float()), torch.from_numpy(eulers)
            return torch.from_numpy(data), torch.from_numpy(eulers)
        else: return data, eulers
    
    
    def load_sample(self, to_tensor=True, create=False, size=100):
        if create: self.sample_data(size)
        data = np.load(f'{self.root}/drm_data_sample.npy')        
        eulers = np.load(f'{self.root}/eulers_sample.npy')
        if to_tensor:
            return normalize_images(torch.from_numpy(data).unsqueeze(1).float()), torch.from_numpy(eulers)
        else: return data, eulers
    
    def sample_data(self, size=100):
        X, Y = self.load_all(to_tensor=False)
        np.save(f'{self.root}/drm_data_sample.npy', X[:size,:size])
        np.save(f'{self.root}/eulers_sample.npy', Y[:size,:size])

    



def test1():
    df = DataFold()
    Xtrain, Ytrain, Xtest, Ytest = df.load_fold()
    print("Xtrain", "Ytrain", "Xtest", "Ytest")
    print(Xtrain.shape, Ytrain.shape, Xtest.shape, Ytest.shape)
    print("min", "max", "mean", "std")
    print(Xtrain.min().item(), Xtrain.max().item(), Xtrain.mean().item(), Xtrain.std().item())

    
def test2():
    df = DataFold()
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = df.load_fold_val()
    print("Xtrain", "Ytrain", "Xval", "Yval", "Xtest", "Ytest")
    print(Xtrain.shape, Ytrain.shape, Xval.shape, Yval.shape, Xtest.shape, Ytest.shape)
    print("min", "max", "mean", "std")
    print(Xtrain.min(), Xtrain.max(), Xtrain.mean(), Xtrain.std())

def test3():
    df = DataFold()
    X, Y = df.load_sample(create=False)
    print("XY", X.shape, Y.shape)


if __name__ == "__main__":
    test1()
    #test2()
    #test3()