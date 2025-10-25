# test.py

import os
import json
import torch
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from data.dataset import SignLanguageDataset
from data.collate import custom_collate_fn
from models.model import IterativeTextGuidedPoseGenerationModel
from configs.config import (
    TEST_FRENCH_FOLDER_2D,
    TEST_GERMAN_FOLDER_2D,
    BATCH_SIZE,
    NUM_WORKERS,
    CHECKPOINT_DIR_LAST,
    CHECKPOINT_DIR_BEST,
    TEST_RESULTS_DIR,
    START_LEARNING_RATE_TEXT_ENCODER,
    START_LEARNING_RATE_GENERATOR,
    TOTAL_STEPS,
    BETA_START,
    BETA_END,
    ETA
)

def main():
    # Define test dataset paths
    test_french_folder_2d = TEST_FRENCH_FOLDER_2D
    test_german_folder_2d = TEST_GERMAN_FOLDER_2D

    # Initialize French and German test datasets
    test_dataset_french_2d = SignLanguageDataset(data_folder=test_french_folder_2d, transform=None)
    test_dataset_german_2d = SignLanguageDataset(data_folder=test_german_folder_2d, transform=None)

    # Concatenate French and German test datasets
    test_dataset = ConcatDataset([test_dataset_french_2d, test_dataset_german_2d])

    # Create DataLoader for the test dataset
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE,  
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )

    # Define the path to the best checkpoint
    best_checkpoint_path = os.path.join(CHECKPOINT_DIR_BEST, 'best-checkpoint.ckpt')  # Ensure the filename matches the saved one

    # Check if the best checkpoint exists
    if not os.path.exists(best_checkpoint_path):
        raise FileNotFoundError(f"Best checkpoint not found at {best_checkpoint_path}")

    # Load the model from the best checkpoint
    model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(
        checkpoint_path=best_checkpoint_path,
        test_results_dir=TEST_RESULTS_DIR,
        text_model_name="xlm-roberta-base",  
        pose_dims=(75, 2),
        hidden_dim=512,      
        num_layers=4,        
        num_heads=8,        
        num_steps=20,        
        lr_text_encoder=START_LEARNING_RATE_TEXT_ENCODER,
        lr_generator=START_LEARNING_RATE_GENERATOR,
        beta_start=BETA_START,
        beta_end=BETA_END,
        eta=ETA,
        total_steps=TOTAL_STEPS
    )

    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        devices=1,  
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',  
        strategy=DDPStrategy(find_unused_parameters=True) if torch.cuda.is_available() else None,
        precision=16 if torch.cuda.is_available() else 32,
        enable_progress_bar=True,  
        logger=True,  
    )

    # Run the testing process
    test_results = trainer.test(model, dataloaders=test_loader, verbose=True)

    # Print test results
    print("Test Results:")
    for key, value in test_results[0].items():
        print(f"{key}: {value}")

    # Define the path to save the test report
    test_report_path = os.path.join(TEST_RESULTS_DIR, 'test_report.json')
    
    # Ensure the test results directory exists
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

    # Save the test report as a JSON file
    with open(test_report_path, 'w') as f:
        json.dump(test_results[0], f, indent=4)
    
    print(f"Test report saved to {test_report_path}")

if __name__ == "__main__":
    main()
