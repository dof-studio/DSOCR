# Project DSOCR
#
# dsocr_prim_test.py
# Feature Tests, do not run this
# by dof-studio/Nathmath
# Open Source Under Apache 2.0 License
# Website: https://github.com/dof-studio/DSOCR

import os
import torch

from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers.utils.bitsandbytes import replace_with_bnb_linear

from ds.modeling_deepseekocr import DeepseekOCRForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'D:/LLM/DeepSeek-OCR/'

raise RuntimeError("This Module WILL NEVER be called!")

# Auto Load
# 
# Without Quant
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    trust_remote_code=True, use_safetensors=True)
model = model.eval().to(dtype=torch.bfloat16, device="cuda")

# Custom Load
#
# With 4B Quantization
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = DeepseekOCRForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ),
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().to("cuda") # .to(torch.bfloat16)

# Custom Load
#
# With 8B Quantization
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = DeepseekOCRForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    quantization_config=BitsAndBytesConfig(
        load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        llm_int8_skip_modules=["sam_model", "vision_model"]
    ),
    trust_remote_code=True,
    use_safetensors=True
)
# model = model.eval().to("cuda") # .to(torch.bfloat16)

# Custom Load using CPU (experimental)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = DeepseekOCRForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().to(device="cpu")

# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'D:/ToolsPy/DSOCR/DeepSeekOCRer/wd/i/desc_ocr_2.jpg'
output_path = 'D:/ToolsPy/DSOCR/DeepSeekOCRer/wd/o'

# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

# Native Inference API (STRONGLY NOT RECOMMENDED)
res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, 
                  base_size = 1024, image_size = 640, crop_mode=True, 
                  save_results = True, test_compress = False)

# Recommend to use our manual inference API
_ = r"""
from dsocr_custom_infer import dsocr_custom_infer_
orig, out, a, b = dsocr_custom_infer_(model, tokenizer, 
                                      prompt=prompt, image_file=image_file, output_path = output_path, 
                                      base_size = 1024, image_size = 640, crop_mode=True, 
                                      save_results = True, test_compress = False)
"""