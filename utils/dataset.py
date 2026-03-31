from torch.utils.data import Dataset, DataLoader

class PerturbationDataset(Dataset):
    """
    单细胞扰动数据集
    """
    def __init__(self, rna_tensors, perturb_tensors, label_tensors):
        """
        Args:
            rna_tensors (torch.Tensor): 表达矩阵
            perturb_tensors (torch.Tensor): 扰动 ID
            label_tensors (torch.Tensor): 标签 (1=匹配, 0=不匹配)
        """
        self.rna = rna_tensors
        self.perturb = perturb_tensors
        self.label = label_tensors
        
    def __len__(self):
        return len(self.label)
        
    def __getitem__(self, idx):
        return {
            'rna': self.rna[idx],
            'perturb': self.perturb[idx].long(), # 确保是长整型用于 Embedding
            'label': self.label[idx].float()
        }

def get_dataloader(data_dict, batch_size=32, shuffle=True):
    dataset = PerturbationDataset(
        data_dict['rna'], 
        data_dict['perturb'], 
        data_dict['label']
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
