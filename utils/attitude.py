import numpy as np
import torch
import torch.nn.functional as F

DELTA = 1e-10

# Converts a list of tensors to a vector
def vectorize(list_of_tensors):
	orig_shapes, vec = [], []
	with torch.no_grad():
		for tensor in list_of_tensors:
			orig_shapes.append(tensor.shape)
			vec.append(tensor.view(-1))
		vec = torch.cat(vec)
	return orig_shapes, vec

# Project from low_rank back to original gradient dimention
def get_new_grads(low_rank, V, grad_shapes, stats):
	new_grad_vec = low_rank.matmul(V).squeeze()
	new_grad_vec = (new_grad_vec * stats[1]) + stats[0]
	new_norm = new_grad_vec.norm()
	return new_grad_vec, new_norm

# Implementation of Gram-Schmidt orthogonalization
def orthogalize(tensor):
	for i in range(tensor.shape[0]):
		if i == 0:
			tensor[i].div_(tensor[i].norm() + DELTA)
		else:
			proj_ = (tensor[i].unsqueeze(0)).matmul(tensor[:i].t()).matmul(tensor[:i])
			tensor[i] = tensor[i] - proj_
			tensor[i].div_(tensor[i].norm() + DELTA)
	return tensor

# Takes in a gradient vector and the shapes of parameter gradients.
# Converts the gradient vector into a list of per-parameter gradient matrices
def reshape_grad_vector(new_grad_vec, grad_shapes):
	new_grads, cur_pos = [], 0
	for shape in grad_shapes:
		delta_shape = np.prod(shape)
		sub_vec = new_grad_vec[cur_pos: (cur_pos + delta_shape)]
		new_grads.append(sub_vec.reshape(shape))
		cur_pos += delta_shape
	return new_grads

def get_ortho_grad_basis(model, task_name, task_loss, device, num_samples, num_pca_basis):
    '''
    Get the random approximation of primary task gradient basis
    '''
    loss = task_loss

    params = list(model.bert.parameters())

    prod_ = []
    for i in range(num_pca_basis):
        v = torch.normal(mean=0.0, std=1.0, size=(num_samples, ), device=device)
        out = torch.autograd.grad(loss, params, v, retain_graph=True)
        prod_.append(vectorize(out)[1].unsqueeze(1))

    with torch.no_grad():
        prod_ = torch.cat(prod_, dim=1)
        ortho_basis = orthogalize(prod_.t())
    return ortho_basis

def get_low_rank(model, task_name, task_loss, basis):
    # get loss and gradients
    loss = task_loss.mean()
    
    # get grads and projections
    params = list(model.parameters())
    all_grads = list(torch.autograd.grad(loss, params, allow_unused=True))

    for idx, (g_, p_) in enumerate(zip(all_grads, params)):
        if g_ is None:
            all_grads[idx] = torch.zeros_like(p_)
    
    this_grads = []
    for idx_, (name, v) in enumerate(model.named_parameters()):
        if "bert" in name:
            this_grads.append(all_grads[idx_])

    grad_shapes, grad_vec = vectorize(this_grads)
    with torch.no_grad():
        grad_vec = grad_vec.unsqueeze(0)
        low_rank = grad_vec.matmul(basis.t())
    # grads = reshape_grad_vector(grad_vec, grad_shapes)
    return low_rank, all_grads

def get_grads_project(model, task_name, task_loss, basis, target_comp=None,
    eta_tilde = 1, eta_pos = 1, eta_neg = 0):
    # get loss and gradients
    loss = task_loss.mean()
    params = list(model.parameters())
    all_grads = list(torch.autograd.grad(loss, params, allow_unused=True))
    
    for idx, (g_, p_) in enumerate(zip(all_grads, params)):
        if g_ is None:
            all_grads[idx] = torch.zeros_like(p_)

    # extract encoder gradients and project on basis
    this_grads, old_norms = [], []
    for idx_, (name, v) in enumerate(model.named_parameters()):
        if "bert" in name:
            # if all_grads[idx_] is None:
            #     all_grads[idx_] = torch.zeros_like(v)
            this_grads.append(all_grads[idx_])
            with torch.no_grad():
                old_norms.append(all_grads[idx_].norm())
    old_norms = np.array(old_norms)

    grad_shapes, grad_vec = vectorize(this_grads)
    with torch.no_grad():
        grad_vec = grad_vec.unsqueeze(0)
        low_rank = grad_vec.matmul(basis.t())
    mask_pos = ((low_rank * target_comp) > 0.0).float()

    new_grad_pos, pos_norm = get_new_grads(low_rank * mask_pos, basis, grad_shapes, stats=(0.0, 1.0))
    new_grad_neg, neg_norm = get_new_grads(low_rank * (1.0 - mask_pos), basis, grad_shapes, stats=(0.0, 1.0))

    with torch.no_grad():
        out_span = grad_vec.squeeze() - (new_grad_pos + new_grad_neg)
        final_grad = (out_span * eta_tilde) + (new_grad_pos * eta_pos) + \
            (new_grad_neg * eta_neg)
        this_grads = reshape_grad_vector(final_grad, grad_shapes)
        new_norm = final_grad.norm()

    # assign adjusted gradients back
    pca_idx_ = 0
    for idx_, (name, _) in enumerate(model.named_parameters()):
        if "bert" in name:
            all_grads[idx_] = this_grads[pca_idx_]
            pca_idx_ += 1
    with torch.no_grad():
        old_norm = ((old_norms**2).sum()).sqrt()
    return loss, (old_norm, new_norm, pos_norm, neg_norm), all_grads

