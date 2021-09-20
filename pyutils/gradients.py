import warnings

import torch
# based on: https://trixi.readthedocs.io/en/develop/_modules/trixi/util/pytorchutils.html#get_smooth_image_gradient

def get_vanilla_image_gradient(model, inpt, err_fn, abs=False):
    if isinstance(model, torch.nn.Module):
        model.zero_grad()
    inpt = inpt.detach()
    inpt.requires_grad = True

    # output = model(inpt)
    err = err_fn(inpt)
    err.backward()

    grad = inpt.grad.detach()

    if isinstance(model, torch.nn.Module):
        model.zero_grad()

    if abs:
        grad = torch.abs(grad)
    return grad.detach()

def get_guided_image_gradient(model: torch.nn.Module, inpt, err_fn, abs=False):
    def guided_relu_hook_function(module, grad_in, grad_out):
        if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU)):
            return (torch.clamp(grad_in[0], min=0.0),)

    model.zero_grad()

    ### Apply hooks
    hook_ids = []
    for mod in model.modules():
        hook_id = mod.register_backward_hook(guided_relu_hook_function)
        hook_ids.append(hook_id)

    inpt = inpt.detach()
    inpt.requires_grad = True

    # output = model(inpt)

    err = err_fn(inpt)
    err.backward()

    grad = inpt.grad.detach()

    model.zero_grad()
    for hooks in hook_ids:
        hooks.remove()

    if abs:
        grad = torch.abs(grad)
    return grad.detach()

def get_smooth_image_gradient(model, inpt, err_fn, abs=True, n_runs=20, eps=0.1,  grad_type="vanilla"):
    grads = []
    for i in range(n_runs):
        inpt = inpt + torch.randn(inpt.size()).to(inpt.device) * eps
        # Take the absolute value only AFTER the mean since gradients could have opposing directions
        if grad_type == "vanilla":
            single_grad = get_vanilla_image_gradient(model, inpt, err_fn, abs=False)
        elif grad_type == "guided":
            single_grad = get_guided_image_gradient(model, inpt, err_fn, abs=False)
        else:
            warnings.warn("This grad_type is not implemented yet")
            single_grad = torch.zeros_like(inpt)
        grads.append(single_grad)

    grad = torch.mean(torch.stack(grads), dim=0)
    if abs:
        grad = torch.abs(grad)
    return grad.detach()

def get_image_gradient(model, inpt, err_fn, abs=True, n_runs=20, eps=0.1,  grad_type="vanilla", smooth=False):
    if smooth:
        grad = get_smooth_image_gradient(model, inpt, err_fn, abs=abs, n_runs=n_runs, eps=eps,  grad_type=grad_type)
    elif grad_type == 'vanilla':
        grad= get_vanilla_image_gradient(model, inpt, err_fn, abs=abs)
    elif grad_type == 'guided':
        grad= get_guided_image_gradient(model, inpt, err_fn, abs=abs)
    else:
        raise NotImplementedError
    return grad


def iterative_gradient_vanillia(model, inpt, err_fn, n_iter=500, step_s=1e-2, get_steps=tuple(),**kwargs):
    out = inpt.clone().detach()
    inter_steps = []
    for i in range(n_iter):
        grad = get_vanilla_image_gradient(model, out, err_fn, abs=False)
        out = out - step_s * grad 
        if i+1 in get_steps:
            inter_steps.append(out.detach().clone().cpu())
    
    if isinstance(model, torch.nn.Module):
            model.zero_grad()
        
    if len(get_steps) > 0:
        return out, inter_steps
    return out

def iterative_gradient_optimizer(model, inpt, err_fn, optimizer=None ,n_iter=500, step_s=1e-2, get_steps=tuple(), **kwargs):
    out = inpt.clone().detach()
    out.requires_grad = True
    optim = optimizer([out], lr=step_s)
    inter_steps = []

    # def closure():
    #     if isinstance(model, torch.nn.Module):
    #         model.zero_grad()
    #     optim.zero_grad()
    #     loss = err_fn(out)
    #     loss.backward()
    #     return loss

    for i in range(n_iter):
        if isinstance(model, torch.nn.Module):
            model.zero_grad()
        optim.zero_grad()
        loss = err_fn(out)
        loss.backward()
        optim.step()
        if i+1 in get_steps:
            inter_steps.append(out.detach().clone().cpu())

    if isinstance(model, torch.nn.Module):
        model.zero_grad()

    if len(get_steps) > 0:
        return out, inter_steps
    return out
        
        



def iterative_gradient(model, inpt, err_fn, optimizer=None ,n_iter=500, step_s=1e-2, get_steps=tuple(), **kwargs):
    if optimizer is None:
        return iterative_gradient_vanillia(model, inpt, err_fn, n_iter, step_s, get_steps, **kwargs)
    else:
        return iterative_gradient_optimizer(model, inpt, err_fn, optimizer ,n_iter, step_s, get_steps, **kwargs)

