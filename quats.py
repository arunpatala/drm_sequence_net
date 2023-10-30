import torch
from scipy.spatial.transform import Rotation
import numpy as np
from utils import DIGITS, DIGITS10, shift_euler, device


QUAT_SYMMETRIES = torch.Tensor([[ 0.0000,  0.0000,  0.0000,  1.0000],
        [ 0.5000,  0.5000,  0.5000,  0.5000],
        [ 0.5000,  0.5000,  0.5000, -0.5000],
        [ 0.5000, -0.5000, -0.5000, -0.5000],
        [ 0.5000, -0.5000,  0.5000,  0.5000],
        [ 0.5000,  0.5000, -0.5000,  0.5000],
        [ 0.5000,  0.5000, -0.5000, -0.5000],
        [ 0.5000, -0.5000, -0.5000,  0.5000],
        [ 0.5000, -0.5000,  0.5000, -0.5000],
        [ 0.0000,  1.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  1.0000,  0.0000],
        [ 1.0000,  0.0000,  0.0000,  0.0000],
        [ 0.7071,  0.0000, -0.7071,  0.0000],
        [ 0.7071,  0.0000,  0.7071,  0.0000],
        [ 0.0000,  0.7071,  0.0000,  0.7071],
        [ 0.0000,  0.7071,  0.0000, -0.7071],
        [ 0.0000,  0.7071, -0.7071,  0.0000],
        [ 0.7071,  0.0000,  0.0000,  0.7071],
        [ 0.7071,  0.0000,  0.0000, -0.7071],
        [ 0.0000,  0.7071,  0.7071,  0.0000],
        [ 0.7071, -0.7071,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.7071, -0.7071],
        [ 0.7071,  0.7071,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.7071,  0.7071]]).float()


class Quaternions:

  @staticmethod        
  def shift(quats, shift):
      eulers = Quaternions.to_eulers(quats)
      eulers = shift_euler(eulers, shift)
      return Quaternions.from_eulersS0(eulers)

  @staticmethod
  def from_eulersS0(eulers, format="ZYZ"):
    return Quaternions.toS0(Quaternions.from_eulers(eulers,format=format))

  @staticmethod
  def from_eulers(eulers, format="ZYZ"):
    rots = Rotation.from_euler(format, eulers.cpu().numpy(), degrees=False)
    rots = rots.as_quat()      
    return torch.from_numpy(rots).to(eulers.device).to(eulers.dtype)

  @staticmethod
  def to_eulers(quats, format='ZYZ'):
    ret = []
    for quat in quats:
        rot = Rotation.from_quat(quat.tolist())
        euler = rot.as_euler(format, degrees=False)
        ret.append(euler.tolist())
    return torch.Tensor(ret).to(quats.device).to(quats.dtype)

  @staticmethod
  def to_rots(quats):
    ret = Rotation.from_quat(quats.cpu()).as_matrix()    
    return torch.from_numpy(ret).to(quats.device)

  @staticmethod
  def multiply(q1, q2):
      q2 = q2.to(q1.device)
      x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
      x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
      w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
      x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
      y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
      z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
      return torch.stack((x, y, z, w), dim=-1)

  @staticmethod
  def multiply_symmetries(quats, QUAT_SYMMETRIES):
    return Quaternions.multiply(quats.unsqueeze(-2), QUAT_SYMMETRIES.unsqueeze(0))

  @staticmethod
  def misorientations(q1, q2, degrees=False, min=True):
    dot = Quaternions.dot(q1, q2)
    angles = 2 * torch.acos(dot)
    if degrees: angles = torch.rad2deg(angles)
    if min: angles = angles.min(-1).values
    return angles
  
  @staticmethod
  def dot(q1, q2):    
    dot = torch.sum(q1*q2.to(q1.device), dim=-1)
    dot = torch.abs(torch.clamp(dot, -1.0, 1.0))
    return dot

  @staticmethod  
  def normalize(q):
      norm = torch.norm(q, dim=-1, keepdim=True)
      return q / norm

  @staticmethod  
  def norm(q):
    return torch.norm(q, dim=-1, keepdim=True)

  @staticmethod  
  def inverse(q):
      q_conj = q.clone()
      q_conj[..., :3] *= -1
      return Quaternions.normalize(q_conj)

  @staticmethod
  def diff(q1, q2):    
    q_diff = Quaternions.multiply(Quaternions.inverse(q1), q2)    
    return q_diff
    
  @staticmethod
  def misorientations_symmetries(q1, q2, symmetries=None, degrees=False, min=True):
    q1s = Quaternions.multiply_symmetries(q1, symmetries or QUAT_SYMMETRIES)
    angles = Quaternions.misorientations(q1s, q2.unsqueeze(-2), degrees, min)
    return angles
  
  @staticmethod
  def toS0(Yquats):
      Yquats_symms = Quaternions.multiply_symmetries(Yquats, QUAT_SYMMETRIES)
      S0 = QUAT_SYMMETRIES[0]
      angles = Quaternions.misorientations(Yquats_symms, S0.unsqueeze(0).unsqueeze(0), degrees=True, min=False)
      indices = (angles.argmin(-1))
      Yquats_syms0 = Yquats_symms[torch.arange(len(Yquats_symms)),indices]
      return Yquats_syms0
  
  @staticmethod    
  def euler_to_strings(eulers, vocab):
      quats = Quaternions.from_eulersS0(eulers)
      return Quaternions.to_strings(quats, vocab)  

  @staticmethod       
  def to_strs(quats, vocab):
      quats_tensor = quats
      rounded_quats = torch.round((quats_tensor + 1) * DIGITS10 / 2.0 )
      #TODO: rounded_quats[rounded_quats==1000] = 999
      rounded_quats = torch.where(rounded_quats == DIGITS10, torch.tensor(DIGITS10-1, dtype=rounded_quats.dtype), rounded_quats)
      rounded_quats_list = rounded_quats.tolist()
      quats_str = [vocab.to_list( '[' + ''.join([str(int(i)).rjust(3, '0') for i in quat]) + ']') for quat in (rounded_quats_list)]
      return torch.Tensor(quats_str).long() 
  
  @staticmethod       
  def to_strs_fast(quats, vocab):
      quats_tensor = quats
      rounded_quats = torch.round((quats_tensor + 1) * DIGITS10 / 2.0 )
      rounded_quats[rounded_quats==1000] = 999
      input_tensor = rounded_quats
      # Assuming input_tensor is your input Nx4 tensor
      # Create tensors to hold each digit (units, tens, hundreds)
      units = input_tensor % 10
      tens = (input_tensor // 10) % 10
      hundreds = (input_tensor // 100)
      output_tensor = torch.stack([hundreds, tens, units], dim=-1)
      t = output_tensor.reshape(-1, 12).long()       
      return vocab.encode(t)

  @staticmethod 
  def from_strs(strs, vocab):
      #TODO: check numeric and len 12 and remove [
      def get_ints(squat):
        try:
          return [int(squat[DIGITS*i:DIGITS*i+DIGITS]) for i in range(4)]
        except: return [0,0,0,1]
      quats_list = [vocab.tensor_to_str(s) for s in strs]
      floats = torch.LongTensor([get_ints(squat) for squat in quats_list])    
      quats = (floats/DIGITS10)*2.0-1.0    
      return Quaternions.normalize(quats)
  
  
  @staticmethod 
  def from_strs_fast(encoded, vocab, normalize=True, strip_begin=True, strip_end=True):
      if strip_begin: encoded = encoded[:, 1:]
      if strip_end: encoded = encoded[:, :-1]
      encoded = encoded - vocab.find("0")
      encoded = encoded.reshape(-1, 4, DIGITS)
      hundreds, tens, units = torch.unbind(encoded, dim=-1)
      tensor = hundreds * 100 + tens * 10 + units
      floats = tensor.view(-1, 4).float().to(encoded.device)
      quats = (floats/DIGITS10)*2.0-1.0
      return Quaternions.normalize(quats) if normalize else quats

  @staticmethod 
  def misQ(quats1, quats2=None):
    quats2 = quats2 or QUAT_SYMMETRIES[0].unsqueeze(0).unsqueeze(0).repeat(quats1.shape[0], quats1.shape[1], 1)
    # Reshape the input quaternions to make them compatible with Quaternions.misorientations
    quats1_reshaped = quats1.reshape(-1, 4)
    quats2_reshaped = quats2.reshape(-1, 4)

    # Calculate the misorientations using Quaternions.misorientations
    mis = Quaternions.misorientations(quats1_reshaped, quats2_reshaped, min=False, degrees=True)

    # Reshape the misorientations array back to the original shape of quats1
    mis_reshaped = mis.view(*quats1.shape[:-1])
    return mis_reshaped

  @staticmethod 
  def misQS(quats1, quats2=None):
    quats2 = quats2 or QUAT_SYMMETRIES[0].unsqueeze(0).unsqueeze(0).repeat(quats1.shape[0], quats1.shape[1], 1)
    # Reshape the input quaternions to make them compatible with Quaternions.misorientations
    quats1_reshaped = quats1.reshape(-1, 4)
    quats2_reshaped = quats2.reshape(-1, 4)

    # Calculate the misorientations using Quaternions.misorientations
    mis = Quaternions.misorientations_symmetries(quats1_reshaped, quats2_reshaped, min=True, degrees=True)

    # Reshape the misorientations array back to the original shape of quats1
    mis_reshaped = mis.view(*quats1.shape[:-1])

    return mis_reshaped
  
  @staticmethod
  def misorientations_pairwise(quats, degrees=True, syms=False):
    n = quats.shape[0]
    quatsi = quats.repeat_interleave(n, dim=0)
    quatsj = quats.repeat(n, 1)

    if syms:
       mis = Quaternions.misorientations_symmetries(quatsi, quatsj, degrees=degrees, min=True)
    else:
       mis = Quaternions.misorientations(quatsi, quatsj, degrees=degrees, min=False)
    mis = mis.reshape(len(quats), len(quats))
    return mis


def test_strs_fast():
   from datafold import DataFold
   Xtrain, Ytrain, Xtest, Ytest = DataFold().load_fold()
   QYtest = Quaternions.from_eulersS0(Ytest)
   from vocab import Vocab
   vocab = Vocab(sort=False)
   QYtest_encoded = Quaternions.to_strs(QYtest, vocab)
   print(QYtest_encoded[:3])

   
   FQYtest_encoded = Quaternions.to_strs_fast(QYtest, vocab)
   print(FQYtest_encoded[:3])
   assert torch.equal(QYtest_encoded, FQYtest_encoded)
   print("equal", torch.equal(QYtest_encoded, FQYtest_encoded))

   
def test_strs_fast2():
   from datafold import DataFold
   Xtrain, Ytrain, Xtest, Ytest = DataFold().load_fold()

   from vocab import Vocab
   vocab = Vocab(sort=False)
   print("vocab", vocab)

   QYtest = Quaternions.from_eulersS0(Ytest)
   print("QYtest", QYtest[:3])
   
   EQYtest = Quaternions.to_strs_fast(QYtest, vocab)
   print(EQYtest[:3])
   print("EQYtest", EQYtest.shape)
   QYtest1 = Quaternions.from_strs_fast(EQYtest, vocab)
   print(QYtest1[:3])

   mis = Quaternions.misorientations(QYtest, QYtest1, degrees=True, min=False)
   print(mis[:10])
   print(mis.min(), mis.max(), mis.mean(), mis.std())


   
   

if __name__ == "__main__":
   test_strs_fast2()