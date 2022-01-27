import torch
import torch.nn as nn

def configure_model(model, eps, momentum, reset_stats, no_stats):
    """Configure model for adaptation by test-time normalization."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            # use batch-wise statistics in forward
            m.train()
            # configure epsilon for stability, and momentum for updates
            m.eps = eps
            m.momentum = momentum
            if reset_stats:
                # reset state to estimate test stats without train stats
                m.reset_running_stats()
            if no_stats:
                # disable state entirely and use only batch stats
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model
