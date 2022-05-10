import math
import torch
import time

# turing
bandwidth_GBs = 672*0.92
shared_max_bytes = 64*1024

preds = []
bench = []

iters = 100
for dim in range(1024, 12288-1024, 1024):

    din = 2048*2
    inner = dim
    dout = dim*4

    tile_rows = 128
    tile_cols = 128

    num_tiles_A = ((din+tile_rows-1)//tile_rows)#*((inner+tile_cols-1)//tile_cols)
    #num_tiles_B = ((dout+tile_cols-1)//tile_cols)
    num_tiles_B = ((inner+tile_rows-1)//tile_rows)*((dout+tile_cols-1)//tile_cols)

    #print(num_tiles_A, ' x ', num_tiles_B)

    bytesA = 2
    bytesB = 2
    tile_bytesA = tile_rows*tile_cols*bytesA
    tile_bytesB = tile_rows*tile_cols*bytesB


    assert tile_bytesA+tile_bytesB <= shared_max_bytes, tile_bytesA+tile_bytesB
    #print(f'Shared utilization: {100*float(tile_bytesA+tile_bytesB)/shared_max_bytes}%')

    tile_loads = num_tiles_A*num_tiles_B*tile_bytesA

    total_loads_GB = float(tile_loads)/1024**3
    predicted_s = float(total_loads_GB)/bandwidth_GBs

    print(f'operation: 100 iterations of einsum([2048, 2, {dim}], [{dim}, {dim*4}] -> [2048, 2, {dim*4}])')
    print(f'predicted from bandwith model:  {predicted_s*iters:.4f}s')

    A = torch.rand(din, inner).half().cuda()
    B = torch.rand(inner, dout).half().cuda()

    # warmup
    for i in range(100):
        torch.matmul(A, B)

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        torch.matmul(A, B)
    torch.cuda.synchronize()
    bench.append(time.time() - t0)
    preds.append(predicted_s*iters)
    print(f'pytorch time: {time.time() - t0:.4f}s')
    print('')

preds = torch.tensor(preds)
bench = torch.tensor(bench)

print(preds)
print(bench)
err = torch.abs(preds-bench)
relerr = err/torch.abs(bench)
print(err.mean(), relerr.mean())


