# Torch DDP

What does it do, and what does it cost:
                Model
                  │
       ┌──────────┴──────────┐
       ↓                     ↓
     Rank 0                Rank 1
       │                     │
    forward               forward
       │                     │
     loss                  loss
       │                     │
  backward                backward
       │                     │
    gradients             gradients
       └──────────┬──────────┘
                  │
              all_reduce
                  │
        synchronized gradients
                  │
             optimizer

## The 3 different kinds of distributed scenarios that I'm aware of