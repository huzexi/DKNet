from os import path, cpu_count

__all__ = ['DefaultConfig']


class DefaultConfig:
    # Directory paths
    dir_h5 = './data'                       # HDF5 folder
    dir_workdir = path.join('workdir')      # Working directory

    # Training
    train_batch_sz = 2                      # Batch size
    train_workers = 5                       # Number of workers for Dataloader
    train_ep = 1000                         # Number of epochs
    train_ep_iter = 1000                    # Number of iterations per epoch
    train_ckp_ep = [                        # How often to save a checkpoint
        [(1, 600), 10],
        [(600, train_ep), 5]                # After Epoch 400, make a checkpoint for every 5 epochs
    ]
    train_patch_sz = (32, 32)               # Spatial size of training samples
    train_lr = 1e-4                         # Learning rate
    train_aug = {                           # Data augmentation
        'flip': True,
        'rotate': True,
    }

    # Valid
    valid = True                            # Enable validation when training
    valid_save_best = True                  # Save checkpoints only when the validation result gets better
    valid_force_save_ep = 250               # Saving model periodically
    valid_mp = cpu_count() // 2             # Multiprocessing
    valid_slice_n = 4                       # Number of slices of validation, for memory saving, set to 0 to disable
    valid_slice_offset = 15                 # Number of overlapped pixels between two slices

    # Test
    test_mp = 8                             # Multiprocessing
    test_slice_n = 3                        # Similar to valid_slice_n and test_slice_offset
    test_slice_offset = 15

    # Task properties
    net = 'DKNet'                           # Network name
    dataset = 'Yeung'                       # Dataset name
    sz_a = (8, 8)                           # Angular size
    color = True
    scale = 4                               # Scale of super-resolution
    shave = 15                              # The border pixels will be shaved in validation and testing

    # Functions
    @classmethod
    def get_train_ckp_ep(cls, ep):
        return next((it[1] for it in cls.train_ckp_ep if ep in range(*it[0])), cls.train_ckp_ep[0][1])
