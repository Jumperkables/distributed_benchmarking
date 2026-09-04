# standard imports

# 3rd party imports
from datasets import load_dataset
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import profile, schedule
from torch.utils.data import DataLoader
from torchview import draw_graph
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
BATCH_SIZE = 32
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
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)




def visualise_model():
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)

    graph = draw_graph(
        model,
        input_data=(input_ids, attention_mask),
        device=DEVICE,
        expand_nested=True,
        depth=4,
        roll=True,
    )

    graph.visual_graph.render("transformer_architecture", format="png", cleanup=True, )
    print("Inspect the attention mechanism")
    print("")




if __name__ == "__main__":
    visualise_model()