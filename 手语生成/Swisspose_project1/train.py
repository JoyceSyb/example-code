# train.py

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch  
from torch.utils.data import DataLoader, ConcatDataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataset import SignLanguageDataset
from data.collate import custom_collate_fn
from models.model import IterativeTextGuidedPoseGenerationModel
from configs.config import (
    TRAIN_FRENCH_FOLDER_2D,
    TRAIN_GERMAN_FOLDER_2D,
    TEST_FRENCH_FOLDER_2D, 
    TEST_GERMAN_FOLDER_2D, 
    BATCH_SIZE,
    NUM_WORKERS,
    MAX_EPOCHS,
    TOTAL_STEPS,
    LR_TEXT_ENCODER,
    LR_GENERATOR,
    BETA_START,
    BETA_END,
    ETA,
    CHECKPOINT_DIR_BEST,
    CHECKPOINT_DIR_LAST,
    LOG_DIR,
    LOG_NAME,
    VALIDATION_SPLIT,
    LAMBDA_LENGTH,
    MAX_NUM_FRAMES,  
    MAX_LENGTH_PREDICTION
)

torch.cuda.empty_cache()

def main():
    # Initialize French and German training datasets
    train_dataset_french_2d = SignLanguageDataset(data_folder=TRAIN_FRENCH_FOLDER_2D)
    train_dataset_german_2d = SignLanguageDataset(data_folder=TRAIN_GERMAN_FOLDER_2D)

    # Concatenate French and German training datasets
    combined_train_dataset = ConcatDataset([train_dataset_french_2d, train_dataset_german_2d])

    # Calculate sizes for training and validation splits
    total_train_size = len(combined_train_dataset)
    val_size = int(VALIDATION_SPLIT * total_train_size)
    train_size = total_train_size - val_size

    # Split the combined training dataset into training and validation sets
    train_dataset, val_dataset = random_split(combined_train_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,  
        pin_memory=True,
        drop_last=True,  
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,  
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )
    
  
    # Initialize the model 
    model = IterativeTextGuidedPoseGenerationModel(
        text_model_name="xlm-roberta-base",  # Using XLM-RoBERTa multilingual model
        pose_dims=(75, 2),  
        hidden_dim=768,     
        num_layers=4,         
        num_heads=8,       
        num_steps=20,
        max_num_frames=MAX_LENGTH_PREDICTION,        
        lr_text_encoder=LR_TEXT_ENCODER,  # Separate learning rates
        lr_generator=LR_GENERATOR,
        beta_start=BETA_START,
        beta_end=BETA_END,
        eta=ETA, 
        lambda_length=LAMBDA_LENGTH,
        total_steps=TOTAL_STEPS  # Pass total_steps
    )

    # Define callbacks and logger
    checkpoint_callback_best = ModelCheckpoint(
        monitor='val_loss',
        dirpath=CHECKPOINT_DIR_BEST,  
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    checkpoint_callback_last = ModelCheckpoint(
        save_last=True,
        dirpath=CHECKPOINT_DIR_LAST, 
        filename='last'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize logger
    logger = TensorBoardLogger(LOG_DIR, name=LOG_NAME)

    # Initialize Trainer
    from pytorch_lightning.strategies import DDPStrategy

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        devices=1,  
        accelerator='gpu',  
        strategy=DDPStrategy(find_unused_parameters=True),  
        precision=32,
        enable_progress_bar=True,  
        callbacks=[checkpoint_callback_best, checkpoint_callback_last, early_stopping_callback, lr_monitor],
        gradient_clip_val=1.0, 
        accumulate_grad_batches=2, 
        log_every_n_steps=10,  
        logger=logger
    )

    # Start Training
    latest_checkpoint = None
    last_checkpoint_path = os.path.join(CHECKPOINT_DIR_LAST, 'last.ckpt')
    if os.path.exists(last_checkpoint_path):
        latest_checkpoint = last_checkpoint_path
        print(f"Resuming training from checkpoint: {latest_checkpoint}")

    trainer.fit(model, train_loader, val_loader, ckpt_path=latest_checkpoint)

if __name__ == "__main__":
    main()
