import torch
from tqdm import tqdm, trange

from dataset import DRMDataset
from utils import device
from quats import Quaternions

def test_loss(model, criterion, vocab, 
        X=None, Y=None, ds=None, 
        batch_size=128):
    num_batches = 0
    total_loss = 0    
    dataset = ds or DRMDataset(X, Y, vocab, xwidth=None)
    dataloader = dataset.to_loader(batch_size, shuffle=False)    
    model.eval()
    pbar = tqdm(dataloader, desc=f"Loss")

    with torch.no_grad():
      for x, batch_x, batch_y, shift in pbar:
          # TODO: collate_fn to device
          x = x.to(device)
          shift = shift.to(device)
          batch_x = batch_x.to(device)
          batch_y = batch_y.to(device)

          output = model(x, batch_x, shift)  

          # Compute loss
          loss = criterion(output, batch_y)
          total_loss += loss.item()
          num_batches += 1
          pbar.set_postfix({"Loss": total_loss / num_batches, "loss":loss.item()})
    return total_loss/num_batches


def test_loss_greedy(model, vocab, 
          X=None, Y=None, ds=None,
          batch_size=128, xwidth=None):
    num_batches, total_loss = 0, 0
    dataset = ds or DRMDataset(X, Y, vocab, xwidth=xwidth)
    pbar = tqdm(dataset.to_loader(batch_size, shuffle=False), desc=f"Loss")
    model.eval()
    miss, outputs, inputs = [], [], []

    with torch.no_grad():
      for x, batch_x, batch_y, shift in pbar:
          x = x.to(device)
          shift = shift.to(device)
          batch_x = batch_x.to(device)
          batch_y = batch_y.to(device)
          
          output = model.generate_batch(x, vocab, shift)
          qoutputs = Quaternions.from_strs_fast(output, vocab, strip_begin=False)
          outputs.append(qoutputs)
          qinputs = Quaternions.from_strs_fast(batch_y, vocab, strip_begin=False)
          inputs.append(qinputs)
          
          #print("device", qinputs.device, qoutputs.device)
          mis = Quaternions.misorientations_symmetries(qinputs, qoutputs, degrees=True)
          miss.append(mis.detach().cpu())
          num_batches += 1
    
    mis = torch.cat(miss)
    print("mis", mis.shape, mis.median(), mis.mean())
    return mis, torch.cat(outputs), torch.cat(inputs)



def pairwise_mis(quats):
  n = quats.shape[0]
  quatsi = quats.repeat_interleave(n, dim=0)
  quatsj = quats.repeat(n, 1)

  mis = Quaternions.misorientations_symmetries(quatsi, quatsj, degrees=True)
  mis = mis.reshape(len(quats), len(quats))
  means = mis.mean(-1)
  min_idx = means.argmin()
  return quats[min_idx]


def TTA(Ytest, Ytests):
  QYtest = Quaternions.from_eulersS0(Ytest)
  
  Ytests = Ytests.reshape(72, -1, 4)
  for i in range(72):
      Ytests[i] = Quaternions.shift(Ytests[i], -i)    
  print("Ytests", Ytests.shape)
  medians, means = [], []
  Ypreds = []
  for i in range(len(Ytests)):
    Ypredsi = Ytests[i]
    mis = Quaternions.misorientations_symmetries(QYtest, Ypredsi, degrees=True)
    #print(i, mis.median(), mis.mean())
    medians.append(mis.median())
    means.append(mis.mean())
  
  medians = torch.stack(medians)
  means = torch.stack(means)
  print("average", "median", medians.mean(), "mean", means.mean())

  for i in trange(len(QYtest)):
    Ypreds.append(pairwise_mis(Ytests[:, i]))

  Ypreds = torch.stack(Ypreds)
  mis = Quaternions.misorientations_symmetries(QYtest, Ypreds, degrees=True)
  print(mis.median(), mis.mean())
  return mis
