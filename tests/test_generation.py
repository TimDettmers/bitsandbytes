from itertools import product
import math

import pytest
import torch

from bitsandbytes.cextension import HIP_ENVIRONMENT
from tests.helpers import TRUE_FALSE, describe_dtype, id_formatter

transformers = pytest.importorskip("transformers")


def get_4bit_config():
    return transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def get_model_and_tokenizer(config):
    model_name_or_path, quant_type = config
    bnb_config = get_4bit_config()
    if quant_type == "16bit":
        bnb_config.load_in_4bit = False
    else:
        bnb_config.bnb_4bit_quant_type = quant_type
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        max_memory={0: "48GB"},
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    return model, tokenizer


def get_prompt_for_generation_eval(text, add_roles=True):
    description = (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    if add_roles:
        prompt = f"{description} ### Human: {text} ### Assistant:"
    else:
        prompt = f"{description} {text}"
    return prompt


def generate(model, tokenizer, text, generation_config, prompt_func=get_prompt_for_generation_eval):
    text = prompt_func(text)
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    outputs = model.generate(inputs=inputs["input_ids"], generation_config=generation_config)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


models = ["huggyllama/llama-7b", "bigscience/bloom-1b7"]
dtypes = ["nf4", "fp4"]


@pytest.fixture(scope="session", params=product(models, dtypes))
def model_and_tokenizer(request):
    model, tokenizer = get_model_and_tokenizer(request.param)
    yield request.param, model, tokenizer
    del model


@pytest.mark.skipif(HIP_ENVIRONMENT, reason="this test is not supported on ROCm yet")
@pytest.mark.parametrize("DQ", TRUE_FALSE, ids=id_formatter("dq"))
@pytest.mark.parametrize("inference_kernel", TRUE_FALSE, ids=id_formatter("inference_kernel"))
@pytest.mark.parametrize("dtype", [torch.float16], ids=describe_dtype)
@pytest.mark.slow
def test_pi(requires_cuda, model_and_tokenizer, inference_kernel, DQ, dtype):
    fixture_config, model, tokenizer = model_and_tokenizer

    generation_config = transformers.GenerationConfig(
        max_new_tokens=20,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )
    generation_config.max_new_tokens = 20

    # text = 'Please write down the first 50 digits of pi.'
    # text = get_prompt_for_generation_eval(text)
    # text += ' Sure, here the first 50 digits of pi: 3.14159'
    n_cases = 6
    text = "3.14159"
    if hasattr(model.config, "quantization_config"):
        model.config.quantization_config.bnb_4bit_compute_dtype = dtype
        model.config.quantization_config.bnb_4bit_use_double_quant = DQ

    if not inference_kernel:
        text = [text] * n_cases
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    x = inputs["input_ids"]
    outputs = []
    if inference_kernel:
        for i in range(n_cases):
            output = model.generate(x, generation_config=generation_config)
            textout = tokenizer.decode(output[0], skip_special_tokens=True)
            outputs.append(textout)
    else:
        outputs = model.generate(x, generation_config=generation_config)
        outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    assert len(outputs) == n_cases
    failure_count = 0
    for i in range(n_cases):
        if not outputs[i][: len(str(math.pi))] == str(math.pi):
            failure_count += 1
    failure_max = 2 if fixture_config[0] == "huggyllama/llama-7b" else 4
    if failure_count > failure_max:
        print(math.pi)
        for out in outputs:
            print(out)
        raise ValueError(f"Failure count: {failure_count}/{n_cases}")
