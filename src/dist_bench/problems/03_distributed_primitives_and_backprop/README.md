# Distributed Primitives and Backpropagation

Getting familiar with the primitives from NCCL, and how they come together to manage backpropagation in modern ML.


## The Plan
- Revise:
    - broadcast
    - reduce
    - all_reduce
    - all_gather
    - reduce_scatter
- Figure out how they combine to make collectives
```
        ↓
distributed gradients
        ↓
gradient synchronization
        ↓
DDP
```

Using some baseline, I'll try to establish:

rank-local forward
        ↓
rank-local loss
        ↓
rank-local gradients
        ↓
gradient synchronization
        ↓
identical gradients
        ↓
optimizer.step()

## Primitives
`broadcast()`
- Taught me I still need to instantiate variables to be received in broadcast()
- Guessed uses in ML:
  - I can see broadcast being a naive way to share weight tensors across models. I know we have ring reduce nowadays which does not require a gather scatter endpoint
  - For modern uses, I wonder if optimiser states (momentum etc...) are gathered and broadcast?

`scatter()`
- Seemed rather straightforward syntactically
- Guessed uses in ML:
  - For when multiple nodes carry shards of an entire model's weights, I can see the initial step of loading and distributing model tensors (a list of parameter tensors) being `torch.chunk`'d and then scattered across ranks.
  - I expected modern practices not to every bother with this from one central node. I imagine each individual node carries literally only a shard from disk which it loads itself.

`gather()`
- Pretty straightforward

`all_gather() <- standard, object, and coalesced variants` 
- SOME of these can notably handle variably sized tensors
- `all_gather()` surprised me at first. I assumed you'd be able to pass an empty list, but you have to imitate the sizes of the incoming expected tensors. Allows variably sized tensors.
- `all_gather_single()`
  - This is awesome. All the incoming tensors are gathered into a single tensor object ready to go, rather than just a list. And, according to the examples it looks like depending on the size of the receiving input dummy tensor, I can have the incoming tensor be in concat OR stacked format
  - Love the elegance of this method
- `all_gather_coalesced()`
  - I wondered if coalescence meant something similar to how it does in CUDA, it turns out yeah kinda
  - The docs are warning that in [2.13](https://docs.pytorch.org/docs/2.13/distributed.html#torch.distributed.all_gather_coalesced) shape checking across all nodes isn't checked to allow for maximal performance and that we should be careful for erroneous errors
  - Turns out for the `nccl` backend i'll be using for ML workloads, its not implemented. Which is interesting
- `all_gather_object()`
  - Looks like theres a variation in `collective_utils` which also does type checking on the back end and errors every rank if it fails. In the docs, not in my build.
  - Uses pickle, and will go via the CPU, since its designed for a wide array of objects. So obviously don't call it for GPU tensors for a pointless `GPU <-> CPU` transfer.
  - I ran into strange behaviour, where code wouldnt work without `torch.cuda.set_device()` set explicitly. Even though the objects I created seemed to have nothing to do with the GPU. Strange
- `all_gather_into_tensor()`
  - Deprecated. This and coalesced are clearly implement better functionally by single.

`reduce()`
- Very straightforward after experience with the other varieties of primitives
- Reduction between tensors mutually. i.e. sum all the values between tensors mutually

`all_reduce()` Also very straightforward now. Though interestingly here I experimented with the reduction ops they made, `dist.PREMUL_SUM(0.1)` for example is very handy!

`reduce_scatter()`
- Functionally understanding what its doing was quite easy. But conceptually why this kind of operation is ever useful was quite tricky. I think the clue is in the ML application mentioned:
```
Suppose a model has layers with these parameters:
Layer 1: 1000 parameters
Layer 2: 2000 parameters
Layer 3: 3000 parameters

After backwards, we have gradients corresponding to these 6000
gradient = [g0, g1, ..., g5999]

We could arbitrarily shared these gradients across 3 GPUs:
- GPU0 owns: [g0,    ..., g1999]
- GPU1 owns: [g2000, ..., g3999]
- GPU2 owns: [g4000, ..., g5999]

These shared don't always correspond to layers:
GPU 0: half of Layer 1 + part of Layer 2
GPU 1: remainder of Layer 2 + part of Layer 3
GPU 2: remainder of Layer 3

             gradient shard

GPU 0: [A0 | A1 | A2]
GPU 1: [B0 | B1 | B2]
GPU 2: [C0 | C1 | C2]

all reduce would give:
GPU 0: [A0+B0+C0 | A1+B1+C1 | A2+B2+C2 ]
GPU 1: [A0+B0+C0 | A1+B1+C1 | A2+B2+C2 ]
GPU 2: [A0+B0+C0 | A1+B1+C1 | A2+B2+C2 ]

          ↓ reduce-scatter

GPU 0: [A0+B0+C0]
GPU 1: [A1+B1+C1]
GPU 2: [A2+B2+C2]
```

`all_to_all()`
- Basically a large permutation command without reduction.
- From the 2.13 docs as a reminder:
```
>>> input = torch.arange(4) + rank * 4
>>> input = list(input.chunk(4))
>>> input
[tensor([0]), tensor([1]), tensor([2]), tensor([3])]     # Rank 0
[tensor([4]), tensor([5]), tensor([6]), tensor([7])]     # Rank 1
[tensor([8]), tensor([9]), tensor([10]), tensor([11])]   # Rank 2
[tensor([12]), tensor([13]), tensor([14]), tensor([15])] # Rank 3
>>> output = list(torch.empty([4], dtype=torch.int64).chunk(4))
>>> dist.all_to_all(output, input)
>>> output
[tensor([0]), tensor([4]), tensor([8]), tensor([12])]    # Rank 0
[tensor([1]), tensor([5]), tensor([9]), tensor([13])]    # Rank 1
[tensor([2]), tensor([6]), tensor([10]), tensor([14])]   # Rank 2
[tensor([3]), tensor([7]), tensor([11]), tensor([15])]   # Rank 3
```

## P2P
`send()` and `recv()`
These were quite straightforward. They need to be paired. But super importantly:

**With an NCCL backend, there is no notion of tags**

If you're familiar with MPI this might sound strnage. I'm not an MPI expert but its crazy enough to me. Instead we rely on operation ordering. How strange.



# Minimal NN forward and backprop
TODO:
- Minimal neural network forward pass and backprop
- Make sure that the gradients and optimiser states are the same on both nodes
- Use torch.dist primitives to support it

# Blocking vs Asynchronous
A small exploration here