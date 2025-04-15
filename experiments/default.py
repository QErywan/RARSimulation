lr = 0.1

lr_type = 'const'

num_rounds = 2000

num_clients = 10

error_feedback = False # not implemented

client_batch_size = 128 # Batch size of a client (for both train and test)

test_every = 40

gpu = 0

net = 'ResNet9'

seed = 123

nbits = 1.0 # Number of bits per coordinate for compression scheme

compression_scheme = 'vector_topk' # compression/decompression scheme ['none', 'vector_topk', 'chunk_topk_recompress', 'chunk_topk_single', 'csh', 'cshtopk_actual', 'cshtopk_estimate']

dataset = 'CIFAR10'

data_per_client = 'sequential'

folder = 'ringallreduce'