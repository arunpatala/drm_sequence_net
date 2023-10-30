from tqdm import tqdm
import itertools
from torch.nn.utils import clip_grad_norm_
from tqdm import trange
import os

from loss import CustomLoss
from utils import mkdir, save_data_to_file, device, get_random_samples, pp, ROOT_PATH
from dataset import DRMDataset
from datafold import DataFold
from vocab import Vocab
from aws import upload_file_to_s3, download_file_from_s3
from models import DRMLSTM
from test import test_loss, test_loss_greedy, TTA


from torch.optim.lr_scheduler import StepLR
def train(exp_name, model, vocab, num_epochs, batch_size,
            Xtrain, Ytrain, Xtest, Ytest, Xval=None, Yval=None,
            xwidth=None, factor=1.0, lr=0.001, dropout=0.1):
  # Training loop
  config = {
    "exp_name": exp_name,
    "vocab": str(vocab),
    "batch_size": batch_size, 
    "num_epochs": num_epochs, 
    "xwidth": xwidth, 
    "factor": factor, 
    "lr": lr,
    "dropout":dropout
  }
  criterion = CustomLoss(factor=factor)
  #TODO learning rate
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)#, lr=0.001)
  scheduler = StepLR(optimizer, step_size=num_epochs//3, gamma=0.1)
  assert num_epochs%3==0
  
  EXP_PATH = os.path.join(ROOT_PATH, exp_name)
  mkdir(EXP_PATH)  
  MODEL_PATH = os.path.join(EXP_PATH, "model_epoch_{}.th")
  save_data_to_file(config, EXP_PATH, "config.json")
  upload_file_to_s3(os.path.join(EXP_PATH, "config.json"))
  dataset = DRMDataset(Xtrain, Ytrain, vocab, xwidth=xwidth)
  val_dataset = DRMDataset(Xval, Yval, vocab, xwidth=None)
  test_dataset = DRMDataset(Xtest, Ytest, vocab, xwidth=None)

  best_tta = 90
  metrics = []
  for epoch in trange(num_epochs):
      print("========================================")
      print("------------TRAINING START--------------")      
      num_batches, total_loss = 0, 0
      metric = {}
      metric['epoch'] = epoch
      metric['start_lr'] = scheduler.get_last_lr()[0]
      # Prepare data in batches
      dataloader = dataset.to_loader(batch_size, shuffle=True)
    
      pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
      model.train()
      for x, batch_x, batch_y, shift in pbar:
          # Transfer data to device
          x = x.to(device)
          shift = shift.to(device)
          batch_x = batch_x.to(device)
          batch_y = batch_y.to(device)

          # Forward pass
          optimizer.zero_grad()
          output = model(x, batch_x, shift)
          loss = criterion(output, batch_y)

          # Backward pass and optimization
          loss.backward()
          clip_grad_norm_(model.parameters(), 0.5)
          optimizer.step()
          total_loss += loss.item()
          num_batches += 1

          # Update tqdm progress bar
          pbar.set_postfix({"Loss": total_loss / num_batches})

      scheduler.step()
      metric['end_lr'] = scheduler.get_last_lr()[0]

      # Print average loss for the epoch
      average_loss = total_loss / num_batches
      metric['train_loss'] = average_loss
      print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")
      print("--------TRAINING FINISHED-------------")
      
      
      
      print("========================================")
      print("-------- VAL LOSS -------------")
      average_test_loss = test_loss(model, criterion, vocab, batch_size, ds=val_dataset)
      metric['val_loss'] = average_test_loss
      print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {average_test_loss:.4f}")      
      print("-------- VAL LOSS FINISHED-------------") 
      print()     
      print("-------- VAL MISORIENTATION -------------")            
      mis, outputs, inputs = test_loss_greedy(model, vocab, batch_size, ds=val_dataset)
      metric['val_mis_median'] = mis.median().item()
      metric['val_mis_mean'] = mis.mean().item()
      print("-------- VAL MISORIENTATION FINISHED -------------")            
      print()           
      print("-------- VAL TTA -------------")            
      mis = TTA(Yval, outputs)
      metric['val_tta_mis_median'] = mis.median().item()
      metric['val_tta_mis_mean'] = mis.mean().item()      
      
      print("========================================")
      print("-------- TEST LOSS -------------")
      average_test_loss = test_loss(model, criterion, vocab, batch_size, ds=test_dataset)
      metric['test_loss'] = average_test_loss
      print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {average_test_loss:.4f}")      
      print("-------- TEST LOSS FINISHED-------------") 
      print()     
      print("-------- TEST MISORIENTATION -------------")            
      mis, outputs, inputs = test_loss_greedy(model, vocab, batch_size, ds=test_dataset)
      metric['test_mis_median'] = mis.median().item()
      metric['test_mis_mean'] = mis.mean().item()
      print("-------- TEST MISORIENTATION FINISHED -------------")            
      print()           
      print("-------- TEST TTA -------------")            
      mis = TTA(Ytest, outputs)
      metric['test_tta_mis_median'] = mis.median().item()
      metric['test_tta_mis_mean'] = mis.mean().item()


      current_tta = mis.median().item()
      torch.save(model.state_dict(), MODEL_PATH.format(epoch))
      upload_file_to_s3(MODEL_PATH.format(epoch))
      if current_tta < best_tta:
        best_tta = current_tta            
        torch.save(model.state_dict(), MODEL_PATH.format("best"))
        upload_file_to_s3(MODEL_PATH.format("best"))
      metric['best_tta'] = best_tta
      print("-------- TTA FINISHED -------------")            
      print()   
      
      print("BEST TTA", best_tta)
      pp.pprint(metric)
      metrics.append(metric)
      save_data_to_file(metrics, EXP_PATH, "metrics.json")
      upload_file_to_s3(os.path.join(EXP_PATH, "metrics.json"))
      print("========================================")
      print("=================END====================")
      print() 

import torch



def main1(exp_name, num_epochs = 90, batch_size=128 * 4 *4, 
              load_model_path=None, load_exp_name=None, load_exp_step="best",
              xwidth=None, factor=1.0, lr=0.001, dropout=0.25):
    df = DataFold()
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = df.load_fold_val()
    Xval, Yval = get_random_samples(Xval, Yval, 1000, seed=0)
    # TODO: add validation
    vocab = Vocab()
    print("vocab", vocab)

    model = DRMLSTM(len(vocab), num_layers=4, dropout=dropout).to(device)
    print("MODEL")
    print(model)
    if load_model_path: 
      model.load_state_dict(torch.load(load_model_path), strict=False);
    if load_exp_name:          
      EXP_PATH = os.path.join(ROOT_PATH, load_exp_name)
      mkdir(EXP_PATH)  
      MODEL_PATH = os.path.join(EXP_PATH, "model_epoch_{}.th").format(load_exp_step)
      download_file_from_s3(MODEL_PATH)
      model.load_state_dict(torch.load(MODEL_PATH))

    
    train(exp_name, model, vocab, num_epochs, batch_size, 
              Xtrain, Ytrain, Xtest, Ytest, Xval, Yval,
              xwidth=xwidth, factor=factor, lr=lr, dropout=dropout)

    print("------------------")
    print("Training finished!")



import sys


if __name__ == "__main__":

    #test_strs_to_quats()
    #print(abc)
    model_path = 'weighted_quatLSTM3_TTA35.th'
    load_exp_name = 'TEST/TESTS/new_45epochs'
    """
    main1(sys.argv[1], 
            num_epochs = 3 * 5, 
            batch_size = 32 * 4 * 4 * 4,
            load_model_path = None, 
            load_exp_name = None, 
            load_exp_step = "best", 
            xwidth = None, 
            factor = 2.0, 
            lr = 0.001, 
            dropout = 0.1)
    """
    
    main1(sys.argv[1], 
            num_epochs = 3 * 15, 
            batch_size = 32 * 4 * 4,
            load_model_path = None, 
            load_exp_name = None, 
            load_exp_step = "best", 
            xwidth = None, 
            factor = 2.0, 
            lr = 0.001, 
            dropout = 0.1)
    
    
    """
    main1(sys.argv[1], 
            num_epochs = 3 , 
            batch_size = 32 * 4 * 4,
            load_model_path = None, 
            load_exp_name = load_exp_name, 
            load_exp_step = "best", 
            xwidth = 1, 
            factor = 2.0, 
            lr = 0.0, 
            dropout = 0.1)
    """
    
    #main1(60, aug=True, load_model_path="drm_quatLSTM_aug.th")

    #"save_drm_quatLSTM.th"
    #test1(model_path)
    #test1_greedy(model_path)
    #test2(model_path)
    #test3()
    #test4("drm_quatLSTM_aug.th")
    #test5_batch(model_path)
    #test6()
    #test7()
    #test8(model_path)

    #test_pairwise()
    #test_weighted_loss1(model_path)
