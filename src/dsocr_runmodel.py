# Project DSOCR
#
# dsocr_runmodel.py
# Providing A Class to Run the DS OCR Model
# by dof-studio/Nathmath
# Open Source Under Apache 2.0 License
# Website: https://github.com/dof-studio/DSOCR


import os
import torch
from typing import Optional, Tuple, Dict, Any
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers.utils.bitsandbytes import replace_with_bnb_linear

# Modified DeepSeek Model for compatible dtype
from ds.modeling_deepseekocr import DeepseekOCRForCausalLM

# Custom Text Accumulator
from dsocr_custom_infer import TextAccumulator

# Custom Inference API
from dsocr_custom_infer import dsocr_custom_infer_


class DeepSeekOCRModel:
    """
    Wrapper around a DeepSeek OCR model providing a reusable loader and an infer() method.
    """

    # Mapping of template name -> (base_size, image_size, crop_mode)
    TEMPLATE_MAP: Dict[str, Tuple[int, int, bool]] = {
        "tiny": (512, 512, False),
        "small": (640, 640, False),
        "base": (1024, 1024, False),
        "large": (1280, 1280, False),
        "super": (1024, 640, True),  # Gundam
        "ultra": (1280, 1024, True), # Gundam + 
        "experimental": (1280, 640, True),
        "default": (1024, 640, True),
        "默认": (1024, 640, True),   # Default, special mapping
    }

    # Custom Prompts Inferrer
    def __custom_prompts__(self, prompt: str, *, ref: str = ""):
        PROMPT_MAP: Dict[str, str] = {
            # Simple OCR
            "Free OCR": "<image>\nFree OCR. ",
            # Standard OCR
            "Standard OCR": "<image>\n<|grounding|>OCR this image. ",
            # Q and A based on the input image
            "Q&A": f"<image>\n{ref if ref else 'What is this?'} Answer me: ",
            # Q and A based on the input image
            "Q&A Chinese": f"<image>\n{ref if ref else '这是什么?'} 回答我: ",
            # Convert image to position marked markdown
            "Convert to Markdown": "<image>\n<|grounding|>Convert the document to markdown. ",
            # Convert image to unpositional pure markdown
            "Convert to Unpositional Markdown": "<image>\nOCR to markdown format. ",
            # Translate the document into another language
            "Translate into": f"<image>\n<|grounding|>Translate the document into {ref if ref else '中文'}. ",
            # Parse Figure
            "Parse Figure": "<image>\nParse the figure. ",
            # Parse Figure in Chinese
            "Parse Figure in Chinese": "<image>\nParse the figure. 用中文解析图表或者解释图片: ",
            # Describe the Image in Detail
            "Describe Image": "<image>\nDescribe the image in detail. ",
            # Describe the Image in Detail in Chinese
            "Describe Image in Chinese": "<image>\nDescribe the image in detail. 用中文仔细描述图片，包括细节: ",
            # Locate an object by reference
            "Locate Object by Reference": f"<image>\nLocate <|ref|>{ref.strip()}<|/ref|> in the image. ",
            # Identify all objects in the image
            "Identify All Objects": "<image>\nIdentify all objects in the image and output them in bounding boxes. ",
            # Accept any other ones, will be directly used
        }
        prompt = prompt.strip()
        if PROMPT_MAP.get(prompt, None) is None:
            # customized
            return prompt if prompt.startswith("<image>\n") else "<image>\n" + prompt
        else:
            return PROMPT_MAP[prompt]
        
    def __init__(
        self,
        model_path: str = "../model/DeepSeek-OCR/",
        device: Optional[str] = None,
        quant: Optional[str] = "nf4",
        dtype: str = "bfloat16",
        use_safetensors: bool = True,
        trust_remote_code: bool = True,
        set_cuda_visible_devices: Optional[str] = None,
        extra_model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Load tokenizer and model into memory.

        Args:
            model_path: local folder or repo id with model files.
            device: torch device string, e.g. "cuda:0" or "cpu". If None, will choose cuda if available.
            quant: quantization type, for example "nf4" or "bnb_8bit". If None or "bfloat16" then use original version
            dtype: preferred dtype for model weights on device. If CPU, then coersively float32.
            use_safetensors: passed to from_pretrained.
            trust_remote_code: passed to from_pretrained.
            set_cuda_visible_devices: optional string to set CUDA_VISIBLE_DEVICES (e.g. '0').
            extra_model_kwargs: additional kwargs forwarded to AutoModel.from_pretrained.
        """
        
        if set_cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = set_cuda_visible_devices

        self.model_path = model_path
        self.use_safetensors = use_safetensors
        self.trust_remote_code = trust_remote_code
        self.extra_model_kwargs = extra_model_kwargs or {}

        # determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device.lower() if isinstance(device, str) else device)

        # quant resolution
        self.quant = quant
        if self.quant not in ("nf4", "fp4", "int4", "bnb_8bit", "bfloat16", "float16", "float32", "float64", None):
            # If not in selected range, then give a None
            self.quant = None

        # dtype selection - store torch dtype for .to()
        dtype = dtype.lower()
        self.requested_dtype = dtype
        if self.requested_dtype not in ("bfloat16", "float16", "float32", "float64"):
            self.requested_dtype = "bfloat16" # by default

        # placeholders
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModel] = None

        # load the resources
        self._load_resources()

    def _inference_dtype_env(self) -> str:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _infer_final_dtype(self) -> torch.dtype:
        """
        Return torch dtype based on requested and device capability.
        """
        if self.requested_dtype is None:
            self.requested_dtype = torch.float32
        if self.device.type == "cuda":
            # prefer bfloat16 if asked and supported, otherwise float16 or float32
            if self.requested_dtype == "bfloat16":
                # best-effort check for bf16 support (some torch versions expose this)
                bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
                if bf16_supported:
                    return torch.bfloat16
                # fallback to float16 if bf16 unavailable
                return torch.float16
            if self.requested_dtype == "float16":
                return torch.float16
            return torch.float32
        else:
            # CPU: keep float32 (bfloat16/float16 on cpu is usually unsupported)
            return torch.float32

    def _load_resources(self):
        """
        Load tokenizer and model and move to device/dtype.
        """
        
        # load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=self.trust_remote_code
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from {self.model_path}: {e}")

        # load model
        model_kwargs = {"trust_remote_code": self.trust_remote_code, "use_safetensors": self.use_safetensors}
        model_kwargs.update(self.extra_model_kwargs)
        
        # This is the new way. Always use our updated, safer model and inference API
        if self.quant is None or self.quant in ("bfloat16", "float16", "float32"):
            
            # Original weights
            self.quant = None # as is, detect best available
            
        # Update final dtype
        final_dtype = self._infer_final_dtype()
                
        # Final Loading
        if True:
            # Quant weights
            self.bnb_config = None

            # 4 bit, by default our trail
            if self.quant in ("nf4", "fp4", "int4"):
                self.bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.quant,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    # bnb_4bit_use_double_quant=True,
                    #
                    # Never opens this to further reduce precision
                )

            # 8 bit, well, if you manually set it
            elif self.quant == "bnb_8bit":
                self.bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    torch_dtype=final_dtype,
                    # This must be on
                    llm_int8_skip_modules=["sam_model", "vision_model"] 
                    # Must skip this since there will be some errors if not excluded
                    # Probably DS modelling issues... but whatever...
                )

            # Otherwise regard as no quant
            else:
                pass
            
            # Manually create a renewed instance and infer
            if self.bnb_config is not None:
                try:
                    if self.model:
                        del self.model
                        torch.cuda.empty_cache()
                except:
                    pass
                self.model = DeepseekOCRForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=self.model_path, 
                    quantization_config=self.bnb_config,
                    **model_kwargs
                    )
            else:
                try:
                    if self.model:
                        del self.model
                        torch.cuda.empty_cache()
                except:
                    pass
                self.model = DeepseekOCRForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=self.model_path, 
                    **model_kwargs
                    )
            self.model.eval()
           
            # to(device)
            try:
                # GPU
                if self.device.type == "cuda" or self.device.type.startswith("cuda"):
                    # move to cuda then cast dtype if supported
                    self.model = self.model.to(self.device)
                    # Note, sometimes, for example, for quant models, the .to() operation will fail
                    
                    # If not using 4 bit or 8 bit quant, then to the destination
                    # Note that quantized model are not supported for further type cast
                    if self.quant not in ("nf4", "fp4", "int4", "bnb_8bit"):
                        try:
                            self.model = self.model.to(dtype=final_dtype)
                        except Exception:
                            # can't reliably cast to bfloat16/any, leave as-is
                            pass                        
                # CPU
                else:
                    # cpu (float32 coersively)
                    self.model = self.model.to(self.device, dtype=torch.float32)
            except Exception as e:
                # non-fatal: continue with whatever device/dtype is available
                print(f"Warning: couldn't move/cast model exactly as requested: {e}")

    def _get_final_inference_env(self) -> Tuple[Any, Any]:
        """
        Return the final device string and dtype after loading the resources
        """
        if self.model is None:
            return None, None
        else:
            return self.device.type, self.model.dtype

    def reload_model(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None, # You may update your pref
        quant: Optional[str] = None,
        dtype: str = None,
        use_safetensors: Optional[bool] = None,
        trust_remote_code: Optional[bool] = None,
        extra_model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Reload or switch model/tokenizer. Pass None to keep existing values.
        """
        if model_path is not None:
            self.model_path = model_path
        if use_safetensors is not None:
            self.use_safetensors = use_safetensors
        if trust_remote_code is not None:
            self.trust_remote_code = trust_remote_code
        if extra_model_kwargs is not None:
            self.extra_model_kwargs = extra_model_kwargs

        # Update device
        if device is not None:
            self.device = torch.device(device)

        # Update quant resolution
        if quant is not None:
            self.quant = quant
        if self.quant not in ("nf4", "fp4", "int4", "bnb_8bit", "bfloat16", "float16", "float32", "float64", None):
            # If not in selected range, then give a None
            self.quant = None

        # Update dtype selection - store torch dtype for .to()
        if dtype is not None:
            dtype = dtype.lower()
            self.requested_dtype = dtype

        # free memory if possible (helpful when switching large models)
        try:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
        except Exception as in_case_you_dont_have_a_cuda_gpu_haha_poor_man_buy_one:
            pass

        self._load_resources()

    def _resolve_template(self, template: Optional[str], base_size: Optional[int], image_size: Optional[int], crop_mode: Optional[bool]):
        """
        Determine base_size, image_size, crop_mode from template or explicit overrides.
        explicit args override template.
        """
        if template:
            template = template.lower()
            if template in self.TEMPLATE_MAP:
                t_base, t_img, t_crop = self.TEMPLATE_MAP[template]
            else:
                raise ValueError(f"Unknown template '{template}'. Known: {list(self.TEMPLATE_MAP.keys())}")
        else:
            # default to 'base'
            t_base, t_img, t_crop = self.TEMPLATE_MAP["base"]

        # override if explicit sizes provided
        final_base = int(base_size) if base_size is not None else t_base
        final_img = int(image_size) if image_size is not None else t_img
        final_crop = bool(crop_mode) if crop_mode is not None else t_crop

        return final_base, final_img, final_crop

    def to_cpu_inplace(self) -> None:
        """
        Move everything to CPU.
        """
        try:
            self.device = torch.device("cpu")
            self.requested_dtype = torch.float32
            self.quant = None
            self.model = self.model.to(device="cpu", dtype=torch.float32)
        except:
            # Fail to original way
            self._load_resources()

    def infer(
        self,
        prompt: str,
        image_file: str,
        output_path: str,
        ref: Optional[str] = "",
        template: Optional[str] = "super",
        base_size: Optional[int] = None,
        image_size: Optional[int] = None,
        crop_mode: Optional[bool] = None,
        save_results: bool = True,
        test_compress: bool = False,
        eval_mode: bool = False,
        streamer: Optional[Any] = None,
        font_path: Optional[str] = None,
        extra_infer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Run inference ON THE THREAD with the loaded model.

        Args:
            prompt: text prompt/presets passed to model.infer.
            image_file: path to the input image file.
            output_path: where to save results (model's infer handles this in your original snippet).
            ref: reference object in detection mode.
            template: one of 'tiny','small','base','large','super', or any supported (or None).
            base_size: override base_size in template.
            image_size: override image_size in template.
            crop_mode: override crop_mode in template.
            save_results: whether to ask model.infer to save results.
            test_compress: whether to pass test_compress True/False to model.infer.
            eval_mode: whether to accumulate tokens and return.
            streamer: An extra streamer to pass if passed, else use the default streamer.
            font_path: The path to the font file you hope to use in writting to the pdf file.
            extra_infer_kwargs: any additional kwargs forwarded to model.infer.

        Returns:
            Tuple[Tuple[str, Any, Any, Any], TextAccumulator]
                The first Tuple[str, Any, Any, Any] is returned by the custom inference API
                The TextAccumulator is an instance of accumulator with texts
        """
        try:
            x = getattr(self, "model")
        except:
            raise RuntimeError("Model/tokenizer not loaded.")
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model/tokenizer not loaded.")
        
        # Prepare parameters
        final_base, final_img, final_crop = self._resolve_template(template, base_size, image_size, crop_mode)

        # Prepare Prompt
        final_prompt = self.__custom_prompts__(prompt, ref=ref)
        
        # Create an accumulator for streamer if not provided
        if not streamer:
            streamer = TextAccumulator(self.tokenizer, print_=False)
            
        # Call Inference API by default
        res = dsocr_custom_infer_(self.model, self.tokenizer, 
                                  prompt=final_prompt, image_file=image_file, output_path=output_path,
                                  base_size=final_base, image_size=final_img, crop_mode=final_crop, eval_mode=eval_mode,
                                  streamer=streamer, font_path=font_path,
                                  save_results=save_results, test_compress=test_compress, **(extra_infer_kwargs or {}))

        return res, streamer

