import torch
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extend_batch(batch, dataloader, batch_size):
    """
    If the batch size is less than batch_size, extends it with
    data from the dataloader until it reaches the required size.
    Here batch is a tensor.
    Returns the extended batch.
    """
    while batch.shape[0] != batch_size:
        dataloader_iterator = iter(dataloader)
        nw_batch,_ = dict.values(next(dataloader_iterator))
        nw_batch = nw_batch.to(DEVICE)
        if nw_batch.shape[0] + batch.shape[0] > batch_size:
            nw_batch = nw_batch[:batch_size - batch.shape[0]]
        batch = torch.cat([batch, nw_batch], 0)
    return batch
def extend_motion(motion, dataloader, batch_size):
    """
    If the batch size is less than batch_size, extends it with
    data from the dataloader until it reaches the required size.
    Here batch is a tensor.
    Returns the extended batch.
    """
    while motion.shape[0] != batch_size:
        dataloader_iterator = iter(dataloader)
        _, nw_motion = dict.values(next(dataloader_iterator))
        nw_motion = nw_motion.permute(0,2,1).to(DEVICE)

        if nw_motion.shape[0] + motion.shape[0] > batch_size:
            nw_motion = nw_motion[:batch_size - motion.shape[0]]
        motion = torch.cat([motion, nw_motion], 0)
    return motion

def gray2color(video):
    '''The input video size: [16, 9, 64, 64]'''
    video = torch.argmax(video, 1).unsqueeze(1).cpu().numpy() *0.1*255
    video = np.round(video*10)/10
    print(video[0,0,30])
    video_r = copy.deepcopy(video)
    video_g = copy.deepcopy(video)
    video_b = copy.deepcopy(video)

    ################ Wall ######################
    video_r[video_r== 51] = 238 
    video_g[video_g== 51] = 213 
    video_b[video_b== 51] = 183 

    ################ Person ######################
    video_r[video_r==127.5] = 212 
    video_g[video_g==127.5] = 76 
    video_b[video_b==127.5] = 60 

    ################ Floor ######################
    video_r[video_r==178.5] = 128
    video_g[video_g==178.5] = 128 
    video_b[video_b==178.5] = 128 

    ################ Window ######################
    video_r[video_r==102] = 68
    video_g[video_g==102] = 117 
    video_b[video_b==102] = 122 

    ################ ceiling ######################
    video_r[video_r==25.5] = 220
    video_g[video_g==25.5] = 220 
    video_b[video_b==25.5] = 220 

    ################ Door ##########################
    video_r[video_r==127.5] = 69 
    video_g[video_g==127.5] = 42 
    video_b[video_b==127.5] = 61 

    video_rgb = torch.cat([torch.tensor(video_r),torch.tensor(video_g), torch.tensor(video_b)], dim=1)/255
    return video_rgb

# Shifts src_tf dim to dest dim
# i.e. shift_dim(x, 1, -1) would be (b, c, t, h, w) -> (b, t, h, w, c)
def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x

# reshapes tensor start from dim i (inclusive)
# to dim j (exclusive) to the desired shape
# e.g. if x.shape = (b, thw, c) then
# view_range(x, 1, 2, (t, h, w)) returns
# x of shape (b, t, h, w, c)
def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)

    
def tensor_slice(x, begin, size):
    assert all([b >= 0 for b in begin])
    size = [l - b if s == -1 else s
            for s, b, l in zip(size, begin, x.shape)]
    assert all([s >= 0 for s in size])

    slices = [slice(b, b + s) for b, s in zip(begin, size)]
    return x[slices]


import math
import numpy as np
# import skvideo.io
# def save_video_grid(video, fname, nrow=None):
#     b, c, t, h, w = video.shape
#     video = video.permute(0, 2, 3, 4, 1)
#     video = (video.cpu().numpy() * 255).astype('uint8')

#     if nrow is None:
#         nrow = math.ceil(math.sqrt(b))
#     ncol = math.ceil(b / nrow)
#     padding = 1
#     video_grid = np.zeros((t, (padding + h) * nrow + padding,
#                            (padding + w) * ncol + padding, c), dtype='uint8')
#     for i in range(b):
#         r = i // ncol
#         c = i % ncol

#         start_r = (padding + h) * r
#         start_c = (padding + w) * c
#         video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]

#     skvideo.io.vwrite(fname, video_grid, inputdict={'-r': '5'})
#     print('saved videos to', fname)


