import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        
        batches = []
        if train:
            for batch_id in range(1, 6):
                batch = unpickle(os.path.join(base_folder, f"data_batch_{batch_id}"))
                batches.append(batch)
        else:
            batch = unpickle(os.path.join(base_folder, "test_batch"))
            batches.append(batch)
        
        self.X = np.concatenate([batch[b'data'] for batch in batches])
        self.X = np.reshape(self.X, (self.X.shape[0], 3, 32, 32)) / 255
        self.y = np.concatenate([batch[b'labels'] for batch in batches])
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        X_items = self.X[index]
        Y_items = self.y[index]
        if isinstance(index, (slice, np.ndarray)):
            Y_items = np.reshape(Y_items, (Y_items.shape[0]))
            X_items = np.reshape(X_items, (X_items.shape[0], 3, 32, 32))
            for item_idx in range(X_items.shape[0]):
                X_item = X_items[item_idx]
                X_items[item_idx] = self.apply_transforms(X_item)
        else:
            X_items = np.reshape(X_items, (3, 32, 32))
            X_items = self.apply_transforms(X_items)
        return (X_items, Y_items)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION
    
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
