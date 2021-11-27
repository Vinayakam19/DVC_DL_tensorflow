import tensorflow as tf
import os
import time
import joblib
import logging
from src.utils.all_utils import get_timestamp

def create_and_save_tensorboard_callback(callbacks_dir, tensorboard_log_dir):
    """
    Creates a callback that saves the model at tensorboard
    """
    unique_name = get_timestamp('tb_logs')
    tb_running_log_dir = os.path.join(tensorboard_log_dir, unique_name)
    tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir = tb_running_log_dir)
    
    tb_callback_filepath = os.path.join(callbacks_dir, 'tensorboard_callback.cb')
    joblib.dump(tensorboard_callbacks, tb_callback_filepath)
    logging.info('Saved tensorboard callback to {}'.format(tb_callback_filepath))

def create_and_save_checkpoint_callback(callbacks_dir, checkpoint_dir):
    """
    Creates a callback that saves the model at a specified frequency.
    """
    
    checkpoint_file_path = os.path.join(checkpoint_dir, 'ckpt_model.h5')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_file_path, save_weights_only=True, verbose=1)
    ckpt_callback_filepath = os.path.join(callbacks_dir, 'checkpoint_callback.cb')
    joblib.dump(checkpoint_callback, ckpt_callback_filepath)
    logging.info('Saved checkpoint callback to {}'.format(ckpt_callback_filepath))
    
def get_callbacks(callbacks_dir_path):
    
    callback_path = [os.path.join(callbacks_dir_path, bin_file) for bin_file  in os.listdir(callbacks_dir_path) if bin_file.endswith('.cb')]
    
    callbacks = [joblib.load(path) for path in callback_path]
    
    logging.info('Loaded {} callbacks from {}'.format(len(callbacks), callbacks_dir_path))
    
    return callbacks 

