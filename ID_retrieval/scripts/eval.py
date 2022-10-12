import tqdm
from PIL import Image
import torch_fidelity

import torch
import torchvision.transforms.functional as tv_f

from tl2.proj.fvcore import TLCfgNode
from tl2 import tl2_utils
from tl2.proj.fvcore.checkpoint import Checkpointer

from ID_retrieval.models.face_recog_model import Backbone


def main(
      content_dir,
      transfer_root,
      face_model_pkl,
      threshold=None,
      enable_fid=True,
      style_dir=None, # for fid
):
  
  device = 'cuda'
  
  facenet = Backbone().eval().requires_grad_(False).to(device)
  Checkpointer(facenet).load_state_dict_from_file(face_model_pkl)
  
  
  content_path_list = tl2_utils.get_filelist_recursive(content_dir, )
  
  emb_content_list = []
  emb_transfer_list = []
  label_id_list = []
  
  for label_id, content_path in enumerate(tqdm.tqdm(content_path_list, desc=f"{content_dir}")):
    
    img_c_pil = Image.open(content_path)
    img_c_tensor = tv_f.to_tensor(img_c_pil)

    img_c_tensor = (img_c_tensor * 2 - 1).unsqueeze(0).to(device)
    emb_c = facenet(img_c_tensor)
    emb_content_list.append(emb_c)
    
    transfer_dir = f"{transfer_root}/{content_path.stem}"
    transfer_path_list = tl2_utils.get_filelist_recursive(transfer_dir)
    for transfer_path in transfer_path_list:
      img_t_pil = Image.open(transfer_path)
      img_t_tensor = tv_f.to_tensor(img_t_pil)
  
      img_t_tensor = (img_t_tensor * 2 - 1).unsqueeze(0).to(device)
      emb_t = facenet(img_t_tensor)
      emb_transfer_list.append(emb_t)
      label_id_list.append(label_id)
      
      pass
      
  
  emb_content = torch.cat(emb_content_list, dim=0)
  emb_transfer = torch.cat(emb_transfer_list, dim=0)
  label_id = torch.tensor(label_id_list, device=device)

  diff = emb_content.unsqueeze(-1) - emb_transfer.transpose(1, 0).unsqueeze(0)
  dist = torch.sum(torch.pow(diff, 2), dim=1)
  minimum, min_idx = torch.min(dist, dim=0)

  id_retrieval = (min_idx == label_id).float().sum() / len(label_id)
  
  print_str = f"\nID_retrieval (top1): {id_retrieval * 100:.2f}%\n"
  if threshold is not None:
    min_idx[minimum > threshold] = -1  # if no match, set idx to -1
    id_retrieval = (min_idx == label_id).float().sum() / len(label_id)
    print_str += f"ID_retrieval (thresh {threshold}): {id_retrieval * 100:.2f}%\n"
    
  if enable_fid:
    metrics_dict = torch_fidelity.calculate_metrics(
      input1=transfer_root,
      input2=style_dir,
      cuda=True,
      isc=False,
      fid=True,
      kid=False,
      verbose=True,
      samples_find_deep=True
    )
    print_str += f"FID: {metrics_dict['frechet_inception_distance']}"
  
  print(print_str)
  pass





if __name__ == '__main__':
  
  cfg_file = "ID_retrieval/configs/ID_retrieval.yaml"
  command = 'eval'
  
  cfg = TLCfgNode.load_yaml_with_command(cfg_file, command)
  
  main(**cfg)