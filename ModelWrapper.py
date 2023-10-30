import os
import torch
from tqdm import tqdm, trange

from utils import device, mkdir, get_random_samples, ROOT_PATH, disp
from aws import download_file_from_s3
from models import DRMLSTM
from vocab import Vocab
from datafold import DataFold
from dataset import DRMDataset
from quats import Quaternions
from cluster import quat_clustering_max, quat_clustering


def shift_euler(eulers, shift):
    angular_shift =  (2 * torch.pi / 72.0) * shift
    eulers = eulers.clone()
    eulers[:, 0] = eulers[:, 0] + angular_shift
    return eulers 

def shift_quats(quats, shift):
    eulers = Quaternions.to_eulers(quats)
    eulers = shift_euler(eulers, shift)
    return Quaternions.from_eulersS0(eulers)

def tta(preds, truths):
    preds = preds.reshape(72, -1, 4)
    truths = truths.reshape(72, -1, 4)
    shifted_preds = preds.clone()
    shifted_truths = truths.clone()
    for i in range(72):
      shifted_preds[i] = shift_quats(shifted_preds[i], -i)
      shifted_truths[i] = shift_quats(shifted_truths[i], -i)
    return shifted_preds, shifted_truths


def pairwise_mis(quats):
  n = quats.shape[0]
  quatsi = quats.repeat_interleave(n, dim=0)
  quatsj = quats.repeat(n, 1)
  #print("quats", quats.shape, quatsi.shape, quatsj.shape)
  mis = Quaternions.misorientations_symmetries(quatsi, quatsj)
  mis = mis.reshape(len(quats), len(quats))
  means = mis.mean(-1)
  min_idx = means.argmin()
  return quats[min_idx]



def pairwise_mis_new(quats):
   centroid = quat_clustering_max(quats, eps=7.5, min_samples=0)
   return centroid




def get_center(Ypreds, Ytruth):
  min_degs = []
  all_mis_pairs = []
  minY = []
  print("Y", Ytruth.shape, Ypreds.shape)
  Ytruth = Ytruth.reshape(-1,3)
  for idx in trange(len(Ytruth)):
    minY.append(pairwise_mis(Ypreds[:, idx]))
  minY = torch.stack(minY)
  print("Y", Ytruth.shape, Ypreds.shape, minY.shape)
  QYtruth = Quaternions.from_eulersS0(Ytruth)
  min_degs = Quaternions.misorientations_symmetries(minY, QYtruth, degrees=True)
  return min_degs


class ModelWrapper:
   
    def __init__(self, model=None, conditional=1):
      self.vocab = Vocab()
      self.model = model or DRMLSTM(len(self.vocab), num_layers=4, conditional=conditional).to(device)      

    def load_exp(self, load_exp_name, load_exp_step="best"):
      EXP_PATH = os.path.join(ROOT_PATH, load_exp_name)
      mkdir(EXP_PATH)  
      MODEL_PATH = (EXP_PATH +"/model_epoch_{}.th").format(load_exp_step)
      if not os.path.exists(MODEL_PATH):
        download_file_from_s3(MODEL_PATH)
      self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
      return self
    
    def tta_batchXY(self, X, Y, batch_size, xwidth=None):
        dataset = DRMDataset(X, Y, self.vocab, xwidth=xwidth)
        miss, outputs, inputs = [], [], []
        self.model.eval()
        loader = dataset.to_loader(shuffle=False, batch_size=batch_size)
        with torch.no_grad():
            for x, batch_x, batch_y, shift in tqdm(loader):
                x = x.to(device)
                shift = shift.to(device)
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)          
                
                output = self.model.generate_batch(x, self.vocab, shift)
                qoutputs = Quaternions.from_strs_fast(output.reshape(-1, 13), self.vocab, strip_begin=False)
                outputs.append(qoutputs)
          
                qinputs = Quaternions.from_strs_fast(batch_y.reshape(-1, 13), self.vocab, strip_begin=False)
                inputs.append(qinputs)
        preds, truths = torch.cat(outputs), torch.cat(inputs) 
        
        min_degs = Quaternions.misorientations_symmetries(preds, truths, degrees=True)
        print("average mis", min_degs.median(), min_degs.mean())
        preds = preds.reshape(72, -1, 4)
        truths = truths.reshape(72, -1, 4)
        shifted_preds, shifted_truths = tta(preds, truths)
        mis = get_center(shifted_preds, Y)
        print("TTA mis", mis.median(), mis.mean())        
        return mis
        
    
    
    def predict_batch(self, X, batch_size, xwidth=1):       
        Y = torch.zeros(X.shape[0], 3).to(X.device).float()
        dataset = DRMDataset(X, Y, self.vocab, xwidth=xwidth)
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for x, batch_x, batch_y, shift in tqdm(dataset.to_loader(shuffle=False, batch_size=batch_size)):
                x = x.to(device)
                shift = shift.to(device)
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                output = self.model.generate_batch(x, self.vocab, shift)
                output = Quaternions.from_strs_fast(output, self.vocab, strip_begin=False)
                outputs.append(output)
        return torch.cat(outputs)
    
    
    def predict_beams(self, X, Y, beams=16, xwidth=1):
        #Y = torch.zeros(X.shape[0], 3).to(X.device).float()
        dataset = DRMDataset(X, Y, self.vocab, xwidth=xwidth)
        outputs = []
        self.model.eval()
        vocab = Vocab()
        with torch.no_grad():
            for x, batch_x, batch_y, shift in tqdm(dataset.to_loader(shuffle=True, batch_size=1)):
                x = x.to(device)
                shift = shift.to(device) 
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                output = self.model.generate_beams(x, self.vocab, shift, beams=beams)
                output = Quaternions.from_strs_fast(output, self.vocab, strip_begin=False)          
                qinputs = Quaternions.from_strs_fast(batch_y, vocab, strip_begin=False)
                
                #print(output)
                #print("truth", output.shape, qinputs.shape, qinputs[0])
                print("------------------------------------------")
                print(qinputs)
                print("output")
                print(output)
                mis = Quaternions.misorientations_symmetries(qinputs.repeat(beams, 1), output, degrees=True)
                print("mis syms", mis)
                
                mis = Quaternions.misorientations(qinputs.repeat(beams, 1), output, degrees=True, min=False)
                print("mis no syms", mis)
                outputs.append(output)
                centroids, _ = quat_clustering(output, eps=7.5, min_samples=2)
                print("centroids", len(centroids), "\n", centroids)

                mis = Quaternions.misorientations_symmetries(qinputs.repeat(len(centroids), 1), centroids, degrees=True)
                
                print("mis centroids", mis)
        return torch.stack(outputs)
    
    def get_mis(self, X, Y, batch_size, tta=False):
       QYpreds = self.predict_batch(X, batch_size)
       #torch.save(QYpreds, "QYpreds.th")       
       QY = Quaternions.from_eulersS0(Y)
       mis1 = Quaternions.misorientations_symmetries(QYpreds, QY, degrees=True)
       #print("mis1", mis1.median(), mis1.mean())
       mis2 = Quaternions.misorientations(QYpreds, QY, degrees=True, min=False)
       #print("mis2", mis2.median(), mis2.mean())
       idx = (torch.nonzero(mis1!=mis2))
       #print("idx", len(idx))
       #print(idx.squeeze())

       if tta: 
          return self.tta_batchXY(X, Y, batch_size=batch_size)
       return mis1
    
    def get_beams(self, X, Y, beams=16):
       QYpreds = self.predict_beams(X, Y, beams)
       print("QYpreds", QYpreds.shape)
       torch.save(QYpreds, "QYpreds.th")       
       QY = Quaternions.from_eulersS0(Y)
       mis = Quaternions.misorientations_symmetries(QYpreds[:,0], QY, degrees=True)
       return mis

from test import test_loss_greedy      

import matplotlib.pyplot as plt

def save_histogram(values, file_name="histogram.png"):
    """
    Generate a histogram from the given list of values and save it to a file.

    Parameters:
    - values (list): List of numeric values to be plotted in the histogram.
    - file_name (str): The name of the file to save the plot. (e.g., "histogram.png")
    """
    
    # Create the histogram
    plt.hist(values, bins=20, edgecolor='black', alpha=0.7)
    plt.title("Histogram of Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # Save the plot to the specified file
    plt.savefig(file_name)
    plt.close()  # Close the plot

import matplotlib.pyplot as plt
import numpy as np

def compare_histograms(values1, values2, file_name="compare_histograms.png", bins=20, label1="label1", label2="label2"):
    """
    Generate and compare histograms from the given lists of values and save to a file.
    The histograms will be plotted side by side for each bin with a blank space between them.

    Parameters:
    - values1 (list): First list of numeric values to be plotted.
    - values2 (list): Second list of numeric values to be plotted.
    - file_name (str): The name of the file to save the plots (e.g., "compare_histogram.png").
    - bins (int or sequence): Number of histogram bins or a sequence defining bin edges.
    """

    # Define the range for the histograms
    hist_range = (0, 60)
    
    # Create custom bin edges to ensure bars are side by side with a gap
    bin_edges = np.linspace(hist_range[0], hist_range[1], bins+1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + bin_width / 2
    
    # Reduced width for each bar to introduce a gap between bars within each bin
    reduced_width = bin_width * 0.33

    # Create the histograms side by side with a gap
    plt.bar(bin_centers - reduced_width/2, np.histogram(values1, bins=bin_edges)[0], 
            width=reduced_width, align='center', alpha=0.5, label=label1, edgecolor='black')
    plt.bar(bin_centers + reduced_width/2, np.histogram(values2, bins=bin_edges)[0], 
            width=reduced_width, align='center', alpha=0.5, label=label2, edgecolor='black')
    
    plt.title("Comparison of Model misorientations")
    plt.xlabel("Misorientation")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')  # Display a legend to label the two histograms

    # Save the plot to the specified file
    plt.savefig(file_name)
    plt.close()

def test1(load_exp_name='TEST/TESTS/new_45epochs', batch_size=16*4, conditional=1, tta=False):
   md = ModelWrapper(conditional=conditional).load_exp(load_exp_name)      
   print(md.model)

   df = DataFold()
   Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = df.load_fold_val()
   #Xtest, Ytest = Xtest[:50], Ytest[:50]
   #Xtest, Ytest = Xval, Yval
   mis = md.get_mis(Xtest, Ytest, batch_size, tta=tta)
   print("test1 mis", mis.median(), mis.mean())
   save_histogram(mis.cpu().tolist())
   return mis

def test1_compare(load_exp_name='TEST/TESTS/new_45epochs', batch_size=16*4, conditional=1, tta=False):
   md = ModelWrapper(conditional=conditional).load_exp(load_exp_name)      
   print(md.model)

   df = DataFold()
   Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = df.load_fold_val()
   #Xtest, Ytest = Xtest[:50], Ytest[:50]
   #Xtest, Ytest = Xval, Yval
   mis1 = md.get_mis(Xtest, Ytest, batch_size, tta=False)
   torch.save(mis1, "mis.th")
   mis2 = md.get_mis(Xtest, Ytest, batch_size, tta=True)
   compare_histograms(mis1.cpu().tolist(), mis2.cpu().tolist())
   return mis2

def test11(load_exp_name1='TEST/TESTS/new_45epochs', load_exp_name2='SYMS/NO_LSTM', batch_size=16*4):
    df = DataFold()
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = df.load_fold_val()
    #Xtest, Ytest = Xval[:1000], Yval[:1000]
    Xtest, Ytest = Xval, Yval

    md1 = ModelWrapper(conditional=1).load_exp(load_exp_name1) 
    md2 = ModelWrapper(conditional=0).load_exp(load_exp_name2)

    
    mis1 = md1.get_mis(Xtest, Ytest, batch_size) 
    mis2 = md2.get_mis(Xtest, Ytest, batch_size) 

    print("---------------------------------------")
    print("mis1", mis1.median(), mis1.mean())
    print("mis2", mis2.median(), mis2.mean())
    #main1(load_exp_name=load_exp_name, load_exp_step="44")
    #test3('ALL')
    cutoff = 5
    for idx, (i,j) in enumerate(zip(mis1, mis2)):
       if i<cutoff and j>2*cutoff:
          try:
            print("---------------------------------------------------------")
            print(idx, i.item(), j.item())
            md1.predict_beams(Xtest[idx:idx+1], Ytest[idx:idx+1])
            md2.predict_beams(Xtest[idx:idx+1], Ytest[idx:idx+1], beams=4)
            print("-------------------------END-----------------------------")
          except KeyboardInterrupt: raise
          except: pass
    
def plot_hist():
    mis1 =  torch.load("mis.th")
    mis2 =  torch.load("../QUATLOSS/mis.th")
    print("mis1", mis1.shape, "mis2", mis2.shape)
    #compare_histograms(mis1.cpu().tolist(), mis2.cpu().tolist(), label1='RNN', label2="Regression")
    compare_histograms(mis2.cpu().tolist(), mis1.cpu().tolist(), label2='SequenceNet', label1="EulerNet")




def test_beams(load_exp_name='TEST/TESTS/new_45epochs', beams=16):
   md = ModelWrapper().load_exp(load_exp_name)      
   print(md.model)

   df = DataFold()
   Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = df.load_fold_val()
   mis = md.get_beams(Xtest, Ytest, beams)
   print("mis", mis.median(), mis.mean())

    
def test2(load_exp_name='TEST/TESTS/new_45epochs', batch_size=16 * 8):
   md = ModelWrapper().load_exp(load_exp_name)      
   print(md.model)

   df = DataFold()
   Xtest, Ytest = df.load_all(slice=1000)
   QY = Quaternions.from_eulers(Ytest.reshape(-1, 3))
   #QYpreds = torch.load("QYpreds.th")
   #print(Quaternions.misorientations_symmetries(QYpreds, QY, degrees=True).median())
   
   print("Xtest", Xtest.shape, Xtest.dtype, Xtest.float().mean(), Xtest.float().std())
   Xtest = Xtest.reshape(-1, 1, 6, 72)
   print("Xtest", Xtest.shape)   
   Ytest = Ytest.reshape(-1, 3)
   mis = md.get_mis(Xtest, Ytest, batch_size)
   print("mis", mis.median(), mis.mean())

def test3(N=500, batch_size=16):
   
   df = DataFold()
   Xtest, Ytest = df.load_all(slice=None)
   print(Xtest.shape)
   H, W, _, _ = Xtest.shape
   QY = Quaternions.from_eulers(Ytest.reshape(-1, 3))
   QYpreds = torch.load(f"DATA/QYpreds{N}.th", map_location=device).reshape(-1, 4)
   mis = Quaternions.misorientations_symmetries(QYpreds, QY, degrees=True)
   
   print("mis", mis.shape, mis.median(), mis.mean())
   mis = mis.reshape(H, W)
   disp(mis, width=6, height=6, vmin=0, cmap='twilight', vmax=mis.max().item(), save_path=f'DATA/mis{N}.png')

   QYpreds = QYpreds.reshape(H, W, -1)
   mis = Quaternions.misQS(QYpreds)
   disp(mis, width=6, height=6, vmin=0, cmap='twilight', vmax=mis.max().item(), save_path=f'DATA/pred_S0mis{N}.png')

   QY = QY.reshape( H, W, -1)
   mis = Quaternions.misQS(QY)
   disp(mis, width=6, height=6, vmin=0, cmap='twilight', vmax=mis.max().item(), save_path=f'DATA/truth_S0mis{N}.png')




def test4(load_exp_name='TEST/TESTS/new_45epochs', batch_size=16 * 8):
  md = ModelWrapper().load_exp(load_exp_name)      
  print(md.model)


  df = DataFold()
  Xtest, Ytest = df.load_all(slice=None)
  Xtest, Ytest = Xtest[:, :100], Ytest[:,:100]
  print(Xtest.shape)
  QY = Quaternions.from_eulers(Ytest.reshape(-1, 3))
  Xsplits = torch.chunk(Xtest, 6, dim=0)
  QYpreds = []
  for Xsplit in Xsplits:
    print("split", Xsplit.shape, Xsplit.dtype)
    qy = md.predict_batch(Xsplit.reshape(-1, 1, 6, 72), batch_size)
    QYpreds.append(qy.cpu())
  QYpreds = torch.cat(QYpreds)
  mis = Quaternions.misorientations_symmetries(QY, QYpreds, degrees=True)
  print("mis", mis.median(), mis.mean())

  QYpreds = QYpreds.view(Xtest.shape[0], Xtest.shape[1], -1)
  print("QYpreds", QYpreds.shape)
  torch.save(QYpreds, "QYpreds.th")
  


def test5(load_exp_name='TEST/TESTS/new_45epochs', batch_size=16):

   df = DataFold()
   Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = df.load_fold_val()
   print(Xtrain.shape, Xval.shape)
   Xtest, Ytest = df.load_all(slice=None)
   print(Xtest.shape)
   print(Xtest.shape[0]*Xtest.shape[1])
   print(Xtrain.shape[0]*72)
   



if __name__ == "__main__":
    test1_compare(tta=False)
    #plot_hist()
    print(abc)
    #main1("test")
    load_exp_name1 = 'SYMS/LSTM'
    #test1(load_exp_name)
    #main1(load_exp_name=load_exp_name, load_exp_step="44")
    #test3('ALL')
    
    load_exp_name2 = 'SYMS/NO_LSTM'
    #test1(load_exp_name, conditional=0)
    #load_exp_name = 'SYMS/S4'    
    #test_beams(load_exp_name)

    test11(load_exp_name1, load_exp_name2)