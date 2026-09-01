# standard imports
import os
import time

# 3rd party imports
from datasets import load_dataset
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

# local imports


# globals
if os.environ["LOCAL_RANK"] is None:
    DEVICE = torch.device("cuda")
else:
    from dist_bench.common.my_utils import rprint, get_dist_env_info    # Cursed placement of an import lawd 4give me
    HOST, RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT = get_dist_env_info()
    DEVICE = torch.device(f"cuda:{LOCAL_RANK}")

torch.cuda.set_device(DEVICE)
NUM_SAMPLES = 2_000
LEARNING_RATE = 5e-5
EPOCHS = 1
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
BATCH_SIZE = 10
SEQ_LENGTH = 500


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



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', action='store_true', help='single gpu training', default=False)
    parser.add_argument('-m', action='store_true', help='multi gpu training', default=False)
    args = parser.parse_args()

    # one of -s or -m must be set
    if not( args.s ^ args.m ):
        raise ValueError("One of -s or -m must be set")

    if args.s:
        single_gpu()

    if args.m:
        multi_gpu()