# FSDP-QLoRA

FSDP-QLoRA combines data parallelism (FSDP enables sharding model parameters, optimizer states, and gradients across GPUs), 4-bit quantization, and LoRA to train LLMs up to 70B parameters on a dual 24GB GPU system. This technique was released by [Answer.AI](https://www.answer.ai/posts/2024-03-06-fsdp-qlora) in collaboration with bitsandbytes to make training LLMs more efficient and accessible for everyone.

This guide provides a brief guide on how bitsandbytes supports storing quantized weights to enable FSDP-QLoRA, and how to run training with the Hugging Face libraries.

> [!TIP]
> Other changes required for bitsandbytes to support FSDP-QLoRA, such as reconstructing the weights from the quantization metadata and preventing quantizing already quantized weights when they're moved from a CPU to GPU, are documented in this [Pull Request](https://github.com/TimDettmers/bitsandbytes/pull/970) and described in the [Enabling 70B Finetuning on Consumer GPUs](https://www.answer.ai/posts/2024-03-14-fsdp-qlora-deep-dive) blog post. We highly recommend reading these resources for a better understanding of FSDP-QLoRA!

## Quantized data storage

FSDP only supports sharding float data types which can be problematic because quantized weights are typically stored as integer data types (uint8). bitsandbytes doesn't have this problem because it uses `StoreChar` to read and write quantized weights regardless of the data type storage. This makes it simple to add a `quant_storage` parameter to the [`~nn.Linear4bit`] and [`~nn.Params4bit`] classes and set it to `torch.uint8` to maintain backward compatibility with the codebase.

```py
import torch
import bitsandbytes as bnb

model = bnb.nn.Linear4bit(
    input_features,
    output_features,
    quant_type="fp4",
    quant_storage=torch.uint8,
)
```

With the `quant_storage` parameter, you can select any of the FSDP supported data types to shard [`~nn.Linear4bit`] with such as bfloat16, float16 or float32.

## Training

bitsandbytes is deeply integrated with the Hugging Face ecosystem, making it easy to use with libraries like [Transformers](https://hf/co/docs/transformers), [PEFT](https://hf/co/docs/peft), and [TRL](https://hf/co/docs/trl).

Before you begin, make sure you have the latest libraries installed.

```bash
pip install -U bitsandbytes accelerate transformers peft trl
```

> [!TIP]
> PEFT provides a configuration file ([fsdp_config_qlora.yaml](https://github.com/huggingface/peft/blob/main/examples/sft/configs/fsdp_config_qlora.yaml)), launch command ([run_peft_qlora_fsdp.sh](https://github.com/huggingface/peft/blob/main/examples/sft/run_peft_qlora_fsdp.sh)), and training script ([train.py](https://github.com/huggingface/peft/blob/main/examples/sft/train.py)) for FSDP-QLoRA. To learn more, check out the [Use PEFT QLoRA and FSDP for finetuning large models on multiple GPUs](https://huggingface.co/docs/peft/main/en/accelerate/fsdp#use-peft-qlora-and-fsdp-for-finetuning-large-models-on-multiple-gpus) documentation.

The important change that enables FSDP-QLoRA training is the `bnb_4bit_quant_storage` parameter in the [`~transformers.BitsAndBytesConfig`] class. This allows you to set the storage data type of the quantized weights to a float data type.

```py
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_storage=torch.bfloat16,
)
```

Pass the [`~transformers.BitsAndBytesConfig`] to a model to set it up for FSDP-QLoRA. You should set the `torch_dtype` parameter to match `bnb_4bit_quant_storage` so that the [`~nn.Linear4bit`] layers are wrapped identically to the `Linear` layers. If the storage types do not match, then each [`~nn.Linear4bit`] layer is wrapped individually.

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)
```

Configure the [`~peft.LoraConfig`] class for QLoRA training by setting `target_modules="all-linear"`.

```py
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)
```

Now you can pass everything to the [`~trl.SFTTrainer`] for training.

```py
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)
trainer.train()
```

## Resources

To learn more about FSDP and QLoRA, check out the following resources:

- The [AnswerDotAI/fsdp_qlora](https://github.com/AnswerDotAI/fsdp_qlora) repository.
- The introductory [You can now train a 70b language model at home](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html) blog post by Answer.AI.
- For an introduction to FSDP, read the [Introducing PyTorch Fully Sharded Data Parallel (FSDP) API](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api) blog post.
- For more details about QLoRA, take a look at the [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes) blog post.
