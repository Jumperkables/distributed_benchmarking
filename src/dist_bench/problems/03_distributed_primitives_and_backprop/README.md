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