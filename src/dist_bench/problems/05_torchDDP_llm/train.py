# standard imports
from dataclasses import dataclass
import os
import time

# 3rd party imports
from datasets import load_dataset
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, schedule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

# local imports


# globals
from dist_bench.common.my_utils import rprint, get_dist_env_info    # Cursed placement of an import lawd 4give me
HOST, RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT = get_dist_env_info()
DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
torch.cuda.set_device(DEVICE)

NUM_SAMPLES = 2_000
LEARNING_RATE = 5e-5
EPOCHS = 1
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
BATCH_SIZE = 3
SEQ_LENGTH = 1000


# Dataset and model objects
print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

# tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.train()

# dataset
dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1",
    split="train",
)
# remove empty lines
dataset = dataset.filter(lambda example: len(example["text"].strip()) > 0)
dataset = dataset.select(
    range(min(NUM_SAMPLES, len(dataset)))
)


def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=SEQ_LENGTH,
        padding="max_length",
    )


dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset.column_names,
)
dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask"],
)




def single_gpu():
    # Dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )


    # training
    total_tokens = 0
    start_time = time.perf_counter()
    step_time = 0
    step_tokens = 0

    for epoch in range(EPOCHS):
        for step, batch in enumerate(dataloader):

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            # For causal language modelling, labels are the input tokens.
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tokens = input_ids.numel()
            total_tokens += tokens
            step_tokens += tokens

            # logging
            if step % 10 == 0:
                elapsed = time.perf_counter() - start_time
                print(
                    f"epoch={epoch} "
                    f"step={step} "
                    f"loss={loss.item():.4f} "
                    f"tokens={total_tokens} "
                    f"total tokens/sec={total_tokens / elapsed:.1f} "
                    f"current tokens/sec={step_tokens/(time.perf_counter()-step_time):.1f} "
                )
                step_time = time.perf_counter()
                step_tokens = 0
    elapsed = time.perf_counter() - start_time
    print("\nTraining complete")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens/sec: {total_tokens / elapsed:.2f}")



def multi_gpu_profile():
    global model
    dist.init_process_group(backend="nccl")
    model = DDP(model, bucket_cap_mb=1000)
    rprint(model._get_ddp_logging_data())
    #import sys; sys.exit(0)

    # Dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Docs says that shuffle needs to be false when using a distributed sampler
        sampler=DistributedSampler(dataset),
    )

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )


    ##################################################
    # Profiler additions
    prof_schedule = schedule(
        wait=2,
        warmup=2,
        active=4,
        repeat=1,
    )

    with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=prof_schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"./traces_NNodes-{os.environ['NNODES']}_rank-{RANK}_bigBuckets"
            ),
            record_shapes=True,
            profile_memory=True,
    ) as prof:
        ##################################################
        # training
        total_tokens = 0
        start_time = time.perf_counter()
        step_time = 0
        step_tokens = 0
        for epoch in range(EPOCHS):
            for step, batch in enumerate(dataloader):

                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)

                # For causal language modelling, labels are the input tokens.
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )

                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                prof.step()

                tokens = input_ids.numel()
                total_tokens += tokens
                step_tokens += tokens

                # logging
                if step % 10 == 0:
                    elapsed = time.perf_counter() - start_time
                    rprint(
                        f"epoch={epoch} "
                        f"step={step} "
                        f"loss={loss.item():.4f} "
                        f"tokens={total_tokens} "
                        f"total tokens/sec={total_tokens / elapsed:.1f} "
                        f"current tokens/sec={step_tokens/(time.perf_counter()-step_time):.1f} "
                    )
                    step_time = time.perf_counter()
                    step_tokens = 0
        elapsed = time.perf_counter() - start_time
        rprint("\nTraining complete")
        rprint(f"Total time: {elapsed:.2f} seconds")
        rprint(f"Total tokens: {total_tokens}")
        rprint(f"Average tokens/sec: {total_tokens / elapsed:.2f}")
        dist.destroy_process_group()



def multi_gpu():
    #############################################
    #### DDP adaptations
    global model
    dist.init_process_group(backend="nccl")
    model = DDP(model)

    # Dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Docs says that shuffle needs to be false when using a distributed sampler
        sampler=DistributedSampler(dataset),
    )
    #############################################

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # training
    total_tokens = 0
    start_time = time.perf_counter()
    step_time = 0
    step_tokens = 0
    for epoch in range(EPOCHS):
        for step, batch in enumerate(dataloader):

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            # For causal language modelling, labels are the input tokens.
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tokens = input_ids.numel()
            total_tokens += tokens
            step_tokens += tokens

            # logging
            if step % 10 == 0:
                elapsed = time.perf_counter() - start_time
                rprint(
                    f"epoch={epoch} "
                    f"step={step} "
                    f"loss={loss.item():.4f} "
                    f"tokens={total_tokens} "
                    f"total tokens/sec={total_tokens / elapsed:.1f} "
                    f"current tokens/sec={step_tokens/(time.perf_counter()-step_time):.1f} "
                )
                step_time = time.perf_counter()
                step_tokens = 0
    elapsed = time.perf_counter() - start_time
    rprint("\nTraining complete")
    rprint(f"Total time: {elapsed:.2f} seconds")
    rprint(f"Total tokens: {total_tokens}")
    rprint(f"Average tokens/sec: {total_tokens / elapsed:.2f}")
    dist.destroy_process_group()




def weighted_grad_hook(state, bucket):
    grad = bucket.buffer()

    # Scale the gradient by this rank's scale factor
    grad.mul_(state.scale)

    # Sum the now-scaled contributions across all ranks
    work = dist.all_reduce(
        grad,
        op=dist.ReduceOp.SUM,
        async_op=True
    )
    return work.get_future().then(
        lambda future: future.value()[0]
    )

@dataclass
class GradScaleState:
    scale: float


def multi_gpu_uneven_batches():
    dist.init_process_group(backend="nccl")
    ###################################
    # Batch size modifications
    if RANK == 0:
        BATCH_SIZE = 12
    if RANK == 1:
        BATCH_SIZE = 3
    # Get the largest batch size across all ranks and scale the rest according to that
    biggest_bsz = torch.tensor(BATCH_SIZE, device=DEVICE)
    dist.all_reduce(biggest_bsz, op=dist.ReduceOp.MAX)
    bsz_loss_scale = BATCH_SIZE / biggest_bsz.item()
    rprint(f"Bsz of this rank: {BATCH_SIZE}")
    rprint(f"Largest bsz across ranks: {biggest_bsz}")
    rprint(f"Loss scale factor for this rank: {bsz_loss_scale}")
    state = GradScaleState(scale=bsz_loss_scale)

    # Model setup
    global model
    model = DDP(model)
    model.register_comm_hook(state, weighted_grad_hook)


    # Dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Docs says that shuffle needs to be false when using a distributed sampler
        sampler=DistributedSampler(dataset),
    )

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # training
    total_tokens = 0
    start_time = time.perf_counter()
    step_time = 0
    step_tokens = 0
    for epoch in range(EPOCHS):
        for step, batch in enumerate(dataloader):

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            # For causal language modelling, labels are the input tokens.
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tokens = input_ids.numel()
            total_tokens += tokens
            step_tokens += tokens

            # logging
            if step % 10 == 0:
                elapsed = time.perf_counter() - start_time
                rprint(
                    f"epoch={epoch} "
                    f"step={step} "
                    f"loss={loss.item():.4f} "
                    f"tokens={total_tokens} "
                    f"total tokens/sec={total_tokens / elapsed:.1f} "
                    f"current tokens/sec={step_tokens/(time.perf_counter()-step_time):.1f} "
                )
                step_time = time.perf_counter()
                step_tokens = 0
    elapsed = time.perf_counter() - start_time
    rprint("\nTraining complete")
    rprint(f"Total time: {elapsed:.2f} seconds")
    rprint(f"Total tokens: {total_tokens}")
    rprint(f"Average tokens/sec: {total_tokens / elapsed:.2f}")
    dist.destroy_process_group()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', action='store_true', help='single gpu training', default=False)
    parser.add_argument('-m', action='store_true', help='multi gpu training', default=False)
    parser.add_argument('-p', action='store_true', help='multi gpu training with the profiler', default=False)
    parser.add_argument('-u', action='store_true', help='multi gpu training with uneven batches allowed', default=False)
    args = parser.parse_args()

    if args.s:
        single_gpu()

    if args.m:
        multi_gpu()

    if args.u:
        multi_gpu_uneven_batches()

    if args.p:
        multi_gpu_profile()