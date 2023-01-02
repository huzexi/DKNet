__all__ = ['DKNetConfig']


class DKNetConfig:
    n_block = 18        # Number of blocks
    kernel = 'Gamma'    # Kernel type
    dense_a = True      # Angular dense connection
    dense_i = True      # Raw image connection
    chns_feat = 32      # Feature channels
