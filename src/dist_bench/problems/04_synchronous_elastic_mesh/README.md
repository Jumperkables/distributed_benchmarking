# Synchronous, Elastic, and Meshes
Covering some of the last few basic concepts before I jump into bigger workloads

## Synchronous launches
The concepts of synchronous launches, like most things, held more complexity under the surface than the surface implies.

### `NCCL`: No tags for `recv` and `send`?
As an ML engineer I mostly care about `nccl` for CUDA and GPU backend collectives. `torch.dist` seems to have a few different backends for many collectives. Something strange that really jumped out about `nccl` backend was that it didn't allow tags for `recv` and `send`?

My understanding of MPI was that tags are pretty central to a lot of communications. They help with asynchronous commands. So, did this imply that `nccl` commands were all synchronous somehow? Is the utility of the tag simply hidden away from the programmer for this backend?

[This explanation on synchronous and asynchronous collective operations](https://docs.pytorch.org/docs/2.13/distributed.html#synchronous-and-asynchronous-collective-operations) was worth a read.

It basically points out that unlike other backends, the `CUDA` nccl backend is already in some sense inherently asynchronous anyway. Work is sent to the `GPU`, which we wait for in CUDA anyway.

So i figured I'd play around with `is_completed()` and `wait()` from `async_op=True` for `nccl` backend primitives and see what happens for myself.

- `async_op=False` gives no object. Nice
- `async_op=True` gives a work object. I mostly need to just care about `wait()` and `is_completed()`

Looks like a lot of the networking semantics and worries I would need to worry about in other APIs are abstracted away and I mostly just need to care about CUDA and CPU pipelines.


### Accidentally making an operation synchronous when I wanted asynchronous
I was trying to see for myself an intermediate where the work wasn't finished yet, get an average of a tensor as it was still being reduced using the following code:
```py
# Play around with some async ops and see what happens
tensor = torch.randn((1000, 32, 128), device=DEVICE)
big_reduce_handle = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
start = time.time()
while time.time()-start < 0.001:
    print(f"{time.time()-start:.6f} | Avg of tensor {tensor.mean().item():.6f} | Completed?: {big_reduce_handle.is_completed()}")
big_reduce_handle.wait()
print(f"After wait: avg of tensor {tensor.mean().item():.6f} |")
dist.destroy_process_group()
```
- The call of `.item()` might be actually waiting on the reduction before resolving, not what i want thinking