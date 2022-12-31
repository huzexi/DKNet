from os import path, cpu_count

__all__ = ['DefaultConfig']


class DefaultConfig:
    # Directory paths
    dir_h5 = './data'
    dir_tmp = path.join('tmp')
    dir_tmp_train = path.join(dir_tmp, 'train')
    dir_tmp_test = path.join(dir_tmp, 'test')
    dir_ckp = path.join('ckp')

    # Training
    train_batch_sz = 2
    train_ep = 5000
    train_ckp_ep = [
        [(1, 600), 10],
        [(600, train_ep), 5]  # After Epoch 400, make checkpoint for every 5 epochs
    ]
    train_ep_iter = 1000
    train_patch_sz = (32, 32)  # Spatial size of training samples
    train_lr = 1e-4
    train_aug = {
        'flip': True,
        'rotate': True,
    }

    # Valid
    valid = True
    valid_save_best = True
    valid_force_save_ep = 250  # Forcedly saving model periodically
    valid_mp = cpu_count() // 2
    valid_slice_n = 4
    valid_slice_offset = 15

    # Test
    test_mp = 0
    test_slice_n = 3
    test_slice_offset = 15

    # Task properties
    dataset = 'Yeung'
    scale = 4
    shave = 15
    sz_a = (8, 8)
    color = True
    net = 'DKNet'

