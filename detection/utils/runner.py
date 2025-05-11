import time
import torch
from tqdm import tqdm
from torch.utils.data import dataloader

from .dataset import FolderDataset


# class Runner():
#     def __init__(self, model, dataset: FolderDataset):
#         self.model = model.eval().to("cuda:0")
#         self.dataset = dataset
#         self.dataloader = dataloader.DataLoader(
#             dataset, 
#             batch_size=1,
#             shuffle=False
#         ) # DO NOT change ANY options

#     @torch.no_grad()
#     def run(self):
#         print('Detection model inferring ...')
#         predictions = {}

#         start_time = time.time()
#         for name, img in tqdm(zip(self.dataset.get_img_name(), self.dataloader)):
#             img = img.to('cuda:0')
#             import torch.nn as nn
#             outputs = nn.Softmax(dim=1)(self.model(img))
#             pred = 1 - outputs[:, 0]
#             pred = pred.detach().cpu().numpy().squeeze().tolist()
#             predictions[name] = pred
#         end_time = time.time()
        

#         return {"predictions": predictions, "time": end_time - start_time}


class Runner():
    def __init__(self, model, dataset: 'FolderDataset', batch_size=32): # Added batch_size parameter
        """
        Initializes the Runner.

        Args:
            model: The PyTorch model to run inference with.
            dataset: An instance of FolderDataset.
            batch_size (int): The batch size for the DataLoader.
        """
        self.model = model.to("cuda:0") # Set model to evaluation mode and move to GPU
        self.dataset = dataset
        # Initialize DataLoader with the specified batch_size
        # shuffle=False is important to maintain order for mapping predictions to names
        self.dataloader = dataloader.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False, # Keep shuffle=False to match names with predictions correctly
            num_workers=4, # Optional: for faster data loading, adjust as needed
            pin_memory=True # Optional: for faster data transfer to GPU
        )

    @torch.no_grad() # Disable gradient calculations for inference
    def run(self):
        """
        Runs inference on the dataset and returns predictions.

        Returns:
            dict: A dictionary containing:
                "predictions": A dictionary mapping image names to their prediction scores.
                "time": The total inference time in seconds.
        """
        print(f'Detection model inferring with batch_size={self.dataloader.batch_size}...')
        predictions = {}
        all_img_names = self.dataset.get_img_name() # Get all image names from the dataset

        # Ensure image names from get_img_name() are stripped of potential newlines
        # This is crucial for correct dictionary keys.
        # self.img_list in FolderDataset is populated by readlines(), which includes '\n'
        stripped_all_img_names = [name.strip() for name in all_img_names]

        current_idx = 0 # To keep track of the current image index for naming

        start_time = time.time()

        # Iterate over batches from the DataLoader
        for img_batch in tqdm(self.dataloader, desc="Inferring Batches"):
            img_batch = img_batch.to('cuda:0') # Move batch to GPU

            # Perform model inference
            # Using torch.nn.functional.softmax is generally preferred over instantiating nn.Softmax in a loop
            outputs = torch.nn.functional.softmax(self.model(img_batch), dim=1)

            # Calculate prediction scores
            # Assuming class 0 is the one to subtract from 1. Adjust if your classes mean something else.
            # outputs shape: (batch_size, num_classes)
            # pred_batch shape: (batch_size,)
            pred_batch = 1 - outputs[:, 0]

            # Detach from graph, move to CPU, convert to NumPy array, then to list
            pred_batch_list = pred_batch.cpu().numpy().tolist() # .squeeze() is not needed here as pred_batch is 1D

            # Determine the actual number of images in the current batch (can be less than batch_size for the last batch)
            actual_batch_size = img_batch.size(0)

            # Get the corresponding image names for the current batch
            names_for_batch = stripped_all_img_names[current_idx : current_idx + actual_batch_size]

            # Store predictions for each image in the batch
            for name, pred_value in zip(names_for_batch, pred_batch_list):
                predictions[name] = pred_value # name is already stripped

            current_idx += actual_batch_size # Update the index for the next batch

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Inference completed in {total_time:.2f} seconds.")

        return {"predictions": predictions, "time": total_time}
