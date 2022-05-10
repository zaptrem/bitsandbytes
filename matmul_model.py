import math
import torch
import time

# turing
bandwidth_GBs = 672*0.93
shared_max_bytes = 64*1024

preds = []
bench = []

compression_factor = 8
iters = 10
for dim in range(1024, 12288-1024, 1024):

    din = 2048*2
    inner = dim
    dout = dim*4

    tile_rows = 128
    tile_cols = 128

    num_tiles_A = ((din+tile_rows-1)//tile_rows)
    num_tiles_B = ((dout+tile_cols-1)//tile_cols)
    inner_reps = (inner+tile_cols-1)//tile_cols

    #print(num_tiles_A, ' x ', num_tiles_B)

    bytesA = 2
    bytesB = 2
    tile_bytesA = tile_rows*tile_cols*bytesA
    tile_bytesB = tile_rows*tile_cols*bytesB


    assert tile_bytesA+tile_bytesB <= shared_max_bytes, tile_bytesA+tile_bytesB
    #print(f'Shared utilization: {100*float(tile_bytesA+tile_bytesB)/shared_max_bytes}%')

    #tile_loads = num_tiles_A*num_tiles_B*tile_bytesA
    tile_loads = ((num_tiles_A*tile_bytesA) + (num_tiles_A*num_tiles_B*tile_bytesB))*inner_reps

    total_loads_GB = float(tile_loads)/1024**3
    predicted_s = float(total_loads_GB)/bandwidth_GBs

    print(f'operation: {iters} iterations of einsum([2048, 2, {dim}], [{dim}, {dim*4}] -> [2048, 2, {dim*4}])')
    print(f'matmul time bandwidth model:  {predicted_s*iters:.4f}s')
    preds.append(predicted_s*iters)


    #A = torch.rand(din, inner).half().cuda()
    #B = torch.rand(inner, dout).half().cuda()

    ## warmup
    #for i in range(50):
    #    torch.matmul(A, B)

    #torch.cuda.synchronize()
    #t0 = time.time()
    #for i in range(iters):
    #    torch.matmul(A, B)
    #torch.cuda.synchronize()
    #bench.append(time.time() - t0)
    #print(f'pytorch time: {time.time() - t0:.4f}s')
    #print('')


# Random row matmul Algorithm:
# We want to use all our shared memory for the matrix A
# we use read-only cache for matrix B (codebooks)
# we have one codebook per column of B and each codebook is 8-bit or 256 entries
# Turing has 32kb of read-only cache
# This means we can hold in Turning 1024*32/(256*2) = 64 columns per SM
# A can hold 64 kb or 32k elements in fp16. That 32*1024/64 = 512 rows and 64 columns
# Each tile computes 512 * B.shape[1] output elements

    code_books_per_col = int(((dout/compression_factor)+256-1)//256)
    tile_rows = 512
    tile_cols = 64//code_books_per_col

    num_tiles_A = ((din+tile_rows-1)//tile_rows)
    num_tiles_B = code_books_per_col
    inner_reps = (inner+tile_cols-1)//tile_cols

    #print(num_tiles_A, ' x ', num_tiles_B)

    bytesA = 2
    bytesB = 2
    tile_bytesA = tile_rows*tile_cols*bytesA
    tile_bytesB = bytesB*code_books_per_col*tile_cols*256


    assert tile_bytesA <= shared_max_bytes, tile_bytesA
    #print(f'Shared utilization: {100*float(tile_bytesA+tile_bytesB)/shared_max_bytes}%')

    #tile_loads = num_tiles_A*num_tiles_B*tile_bytesA
    tile_loads = ((num_tiles_A*tile_bytesA) + (num_tiles_A*num_tiles_B*tile_bytesB))*inner_reps

    total_loads_GB = float(tile_loads)/1024**3
    predicted_s = float(total_loads_GB)/bandwidth_GBs

    print(f'rdm row matmul time with {compression_factor}x compression:  {predicted_s*iters:.4f}s\t speedup: {preds[-1]/(predicted_s*iters):.4f}x')

# Random tile matmul Algorithm:
# We want to use all our shared memory for the matrix A
# we use read-only cache for matrix B (codebooks)
# with a compression factor of 4x a codebook is used for 256*4=1k elements. Turing read-only cache is of size 32k
# 16k float elements or 16 codebooks. This means a tile-size of 4k x 4k
# flatten out to 64 inner dimension we get 64 x 256 sized tiles

    num_code_books_per_cache = 32*1024/(2*256)
    elements = num_code_books_per_cache*256*compression_factor
    tile_cols_B = (elements+64-1)//64
    tile_rows = 512
    tile_cols = 64

    num_tiles_A = ((din+tile_rows-1)//tile_rows)
    num_tiles_B = ((dout+tile_cols_B-1)//tile_cols_B)
    inner_reps = (inner+tile_cols-1)//tile_cols
    #print(num_code_books_per_cache, elements, tile_cols_B, num_tiles_B)

    #print(num_tiles_A, ' x ', num_tiles_B)

    bytesA = 2
    bytesB = 2
    tile_bytesA = tile_rows*tile_cols*bytesA
    tile_bytesB = tile_cols*tile_cols_B*bytesB


    assert tile_bytesA <= shared_max_bytes, tile_bytesA
    #print(f'Shared utilization: {100*float(tile_bytesA+tile_bytesB)/shared_max_bytes}%')

    #tile_loads = num_tiles_A*num_tiles_B*tile_bytesA
    tile_loads = ((num_tiles_A*tile_bytesA) + (num_tiles_A*num_tiles_B*tile_bytesB))*inner_reps

    total_loads_GB = float(tile_loads)/1024**3
    predicted_s = float(total_loads_GB)/bandwidth_GBs

    print(f'rdm tile matmul time with {compression_factor}x compression:  {predicted_s*iters:.4f}s\t speedup: {preds[-1]/(predicted_s*iters):.4f}x')

preds = torch.tensor(preds)
bench = torch.tensor(bench)

if len(preds) > 0 and len(bench) > 0:
    print(preds)
    print(bench)
    err = torch.abs(preds-bench)
    relerr = err/torch.abs(bench)
    print(err.mean(), relerr.mean())


