'''
Description: 
Autor: Jiachen Sun
Date: 2022-09-03 18:07:28
LastEditors: Jiachen Sun
LastEditTime: 2022-09-05 18:44:25
'''
import torch
import torch.nn as nn
import torch.jit

def reset(model,model_state):
    model.load_state_dict(model_state, strict=True)
    return model

def confidence_select(output,p=0.1):
    size = output.shape[0]
    pass

def augment(input,size):
    pc = input['pc']
    pc = pc.repeat([size,1,1])
    #TODO augmentation happens here
    return {'pc':pc}

def collect_params(model):
    params = model.parameters()
    return params

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    return model

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    # augment
    x = augment(x)
    # forward
    outputs = model(**x)
    # adapt
    loss = softmax_entropy(outputs['logit'].mean(0))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs