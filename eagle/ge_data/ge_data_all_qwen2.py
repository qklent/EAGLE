import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description="sp")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])
    parser.add_argument("--outdir", type=str, default="outdir0")
    return parser.parse_args()


def setup_environment(gpu_index):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)[1:-1]


def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break

    return list1[:prefix_length], prefix_length


def preprocess_conversation(examples, tokenizer):
    new_examples = {"conversation": [], "input_ids": [], "loss_mask": []}

    for i in range(len(examples["id"])):
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            }
        ]
        conv = examples["conversation"][i]
        if conv[0]["role"] != "system":
            conv = messages + conv

        conversation = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.unk_token_id

        input_ids = tokenizer(
            conversation, return_tensors="pt", max_length=2048, add_special_tokens=False
        ).input_ids[0]
        loss_mask = torch.ones_like(input_ids)

        # Process conversation turns
        loss_mask = process_conversation_turns(conversation, tokenizer, loss_mask)

        new_examples["conversation"].append(conversation)
        new_examples["input_ids"].append(input_ids[None, :])
        new_examples["loss_mask"].append(loss_mask[None, :])

    return new_examples


def process_conversation_turns(conversation, tokenizer, loss_mask):
    sep = "<|im_end|>\n<|im_start|>assistant\n"
    sep2 = "<|im_end|>\n<|im_start|>user\n"
    turns = conversation.split(sep2)

    turns[1] = turns[0] + sep2 + turns[1]
    turns = turns[1:]

    cur_len = 1
    loss_mask[:cur_len] = 0

    for i, turn in enumerate(turns):
        if turn == "":
            break
        turn_len = len(tokenizer(turn).input_ids)

        parts = turn.split(sep)
        if len(parts) != 2:
            break

        parts[0] += sep
        instruction_len = len(tokenizer(parts[0]).input_ids)

        if i == 0:
            loss_mask[0 : cur_len + instruction_len - 2] = 0
        else:
            loss_mask[cur_len - 6 : cur_len + instruction_len - 2] = 0
        cur_len += turn_len
        cur_len += 5

    loss_mask[cur_len:] = 0
    return loss_mask


def build_dataset_rank(tokenizer, args, split="train"):
    ds = load_dataset("Vikhrmodels/GrandMaster-PRO-MAX")
    ds = ds["train"]
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))
    original_columns1 = ds1.column_names

    ds1 = ds1.map(
        lambda examples: preprocess_conversation(examples, tokenizer),
        batched=True,
        remove_columns=original_columns1,
        load_from_cache_file=False,
    )

    ds1.set_format(type="torch")
    return ds1


@torch.no_grad()
def generate_embeddings(data, model):
    input_ids = data["input_ids"]
    outs_big = model(input_ids.cuda(), output_hidden_states=True)
    hidden_state_big = outs_big.hidden_states[-1]
    return {
        "input_ids": input_ids.cpu()[0],
        "hidden_state": hidden_state_big.cpu()[0],
        "loss_mask": data["loss_mask"].cpu()[0],
    }


def setup_output_directory(outdir, index):
    output_path = f"{outdir}/{index}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path


def save_data_point(output_dir, data_point):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    current_length = len(os.listdir(output_dir))
    torch.save(data_point, f"{output_dir}/data_{current_length}.ckpt")


def main():
    args = parse_arguments()
    setup_environment(args.gpu_index)

    modelname = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        modelname, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()

    dataset = build_dataset_rank(tokenizer, args)
    output_dir = setup_output_directory(args.outdir, args.index)

    for data in dataset:
        output_data = generate_embeddings(data, model)
        save_data_point(output_dir, output_data)


if __name__ == "__main__":
    main()
