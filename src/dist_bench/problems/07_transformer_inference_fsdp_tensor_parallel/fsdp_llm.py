# standard imports
import time
from unittest import case

# 3rd party imports
from datasets import load_dataset
import torch
from torch.distributed.fsdp import fully_shard, FSDPModule
from torch.utils.data import DataLoader
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
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"#"HuggingFaceTB/SmolLM3-3B"
BATCH_SIZE = 5
SEQ_LENGTH = 1000


# Dataset and model objects
print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

# tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.train()

##################################
# FSDPModule
FSDP = 'Decoder'
#FSDP = 'SubDecoder'

match FSDP:
    case 'Decoder':
        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)
        rprint(f"FSDP: {FSDP}")
    case 'SubDecoder':
        for layer in model.model.layers:
            fully_shard(layer.mlp.up_proj)
            fully_shard(layer.mlp.down_proj)
            fully_shard(layer.mlp)
            fully_shard(layer)
        fully_shard(model)
        rprint(f"FSDP: {FSDP}")
    case _:
        rprint("No FSDP")

layer = model.model.layers[0]
def pre_hook(module, inputs):
    torch.cuda.synchronize()
    rprint(
        f"ENTER layer 0 | "
        f"allocated={torch.cuda.memory_allocated(DEVICE)/1024**3:.2f} GB | "
        f"reserved={torch.cuda.memory_reserved(DEVICE)/1024**3:.2f} GB"
    )


def post_hook(module, inputs, output):
    torch.cuda.synchronize()
    rprint(
        f"EXIT layer 0 | "
        f"allocated={torch.cuda.memory_allocated(DEVICE)/1024**3:.2f} GB | "
        f"reserved={torch.cuda.memory_reserved(DEVICE)/1024**3:.2f} GB"
    )


layer.register_forward_pre_hook(pre_hook)
layer.register_forward_hook(post_hook)
##################################


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
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)


saved_bytes = 0

def pack_hook(tensor):
    global saved_bytes
    saved_bytes += tensor.numel() * tensor.element_size()
    return tensor

def unpack_hook(tensor):
    return tensor


def main():
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

            with torch.autograd.graph.saved_tensors_hooks(
                    pack_hook,
                    unpack_hook,
            ):
                before = torch.cuda.memory_allocated()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                torch.cuda.synchronize()

                peak = torch.cuda.max_memory_allocated()

                loss = outputs.loss

                print("Activations baseline:", before / 2 ** 30, "GB")
                print("Activations peak:    ", peak / 2 ** 30, "GB")
                print("Activations increase:", (peak - before) / 2 ** 30, "GB")

                optimizer.zero_grad()
                print(f"Memory allocated (before backward): {torch.cuda.memory_allocated(DEVICE)/1024**3:.2f} GB")
                print(f"Memory reserved (before backward): {torch.cuda.memory_reserved(DEVICE)/1024**3:.2f} GB")
                print("BACKWARDS BEGINS")
                loss.backward()

            # Measure VRAM burdens
            param_bytes_logical = sum(
                p.numel() * p.element_size()
                for p in model.parameters()
            )
            param_bytes_local = sum(
                p.to_local().numel() * p.element_size()
                for p in model.parameters()
            )
            grad_bytes = sum(
                p.grad.numel() * p.grad.element_size()
                for p in model.parameters()
                if p.grad is not None
            )

            print(f"saved activations: {saved_bytes / 1024 ** 2:.1f} MB")
            print(f"parameters logical: {param_bytes_logical / 1024 ** 2:.1f} MB")
            print(f"parameters local: {param_bytes_local / 1024 ** 2:.1f} MB")
            print(f"gradients:  {grad_bytes / 1024 ** 2:.1f} MB")
            print(f"Memory allocated (after backward): {torch.cuda.memory_allocated(DEVICE)/1024**3:.2f} GB")
            print(f"Memory reserved (after backward): {torch.cuda.memory_reserved(DEVICE)/1024**3:.2f} GB")

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



if __name__ == ("__main__"):
    main()