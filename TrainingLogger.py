import torch
import torchvision
import numpy as np
import pickle

class TrainingDynamicsLogger(object):
    """
    Helper class for saving training dynamics for each iteration.
    Maintain a list containing output probability for each sample.
    """
    def __init__(self, filename=None):
        self.training_dynamics = []

    def log_tuple(self, tuple):
        self.training_dynamics.append(tuple)

    def save_training_dynamics(self, filepath, data_name=None):
        pickled_data = {
            'data-name': data_name,
            'training_dynamics': self.training_dynamics
        }

        with open(filepath, 'wb') as handle:
            pickle.dump(pickled_data, handle)