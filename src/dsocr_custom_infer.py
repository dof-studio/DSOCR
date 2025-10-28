# Project DSOCR
#
# dsocr_custom_infer.py
# Providing Custom Inference API for DSOCR
# by dof-studio/Nathmath
# Open Source Under Apache 2.0 License
# Website: https://github.com/dof-studio/DSOCR

import os
import re
import torch
import math
from tqdm import tqdm
import importlib, inspect
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from typing import Tuple, Any
from transformers import TextStreamer

# Make Sure you are at the correct place
from dsocr_pdf_utils import reconstruct_pdf_from_ocr_blob as reconst
from ds.modeling_deepseekocr import DeepseekOCRForCausalLM
from ds.modeling_deepseekocr import format_messages, load_pil_images, text_encode
from ds.modeling_deepseekocr import dynamic_preprocess, re_match, process_image_with_refs
from ds.modeling_deepseekocr import BasicImageTransform, NoEOSTextStreamer

# Streamer API
class CallbackStreamer(TextStreamer):
    
    def __init__(self, tokenizer, api_callback, skip_prompt=True, skip_special_tokens=False):
        super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)
        self.api_callback = api_callback  
        # Pass in a function to send the result to the external API

    def on_finalized_text(self, text, stream_end=False):
        # Override default method: transformers are printed here by default
        if text:
            # Call back, use keyword args
            self.api_callback(text=text, stream_end=stream_end)  
          
    @staticmethod
    def _sample_api(text, finished):
        print(f"[Send to API] {text}", "(finished)" if finished else "")
        
        
# Simple Accumulator API
class TextAccumulator(TextStreamer):
    
    def __init__(self, tokenizer, print_=False, skip_prompt=True, skip_special_tokens=False):
        super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)
        
        # Print when receving
        self.print_ = print_
        
        # Accumulated Text
        self._accumulated = ""
        
        # Accumulated List
        self._batches = []
        
    def accumulate(self, text, stream_end, **kwargs):
        
        text_ = text.strip()
        self._accumulated += text_
        self._batches.append(text_)
        
    def on_finalized_text(self, text, stream_end=False):
        # Override default method: transformers are printed here by default
        if text:
            self.accumulate(text=text, stream_end=stream_end)  
            if self.print_:
                print(text, sep="", end="\n" if stream_end else "")
        
    @property
    def whole(self) -> str:
        return self._accumulated
    
    @property
    def batches(self) -> list:
        return self._batches
    
    @property
    def len(self) -> int:
        return len(self._accumulated)


# Test device support of NF4
def supports_bnb_nf4_() -> bool:
    """Return True if bitsandbytes supports NF4 quantization, else False."""
    try:
        bnb = importlib.import_module("bitsandbytes")
        if not hasattr(bnb, "nn"):
            return False

        # Look for NF4-related classes
        for cls_name in dir(bnb.nn):
            if "nf4" in cls_name.lower() or "4bit" in cls_name.lower():
                cls = getattr(bnb.nn, cls_name)
                try:
                    sig = inspect.signature(cls)
                    kwargs = {"quant_type": "nf4"} if "quant_type" in sig.parameters else {}
                    _ = cls(4, 4, **kwargs)
                    return True
                except Exception:
                    continue
        return False
    except Exception:
        return False
            

# Test device support of bnb_8bit
def supports_bnb_8bit_() -> bool:
    """Return True if bitsandbytes exposes usable 8-bit classes (e.g. Linear8bitLt), else False."""
    try:
        bnb = importlib.import_module("bitsandbytes")
        if not hasattr(bnb, "nn"):
            return False

        # candidate substrings likely indicating 8-bit support
        candidates = ("8bit", "int8", "linear8", "quant8")

        for cls_name in dir(bnb.nn):
            name_lower = cls_name.lower()
            if any(sub in name_lower for sub in candidates):
                cls = getattr(bnb.nn, cls_name)
                # Skip non-callable attributes
                if not callable(cls):
                    continue
                try:
                    sig = inspect.signature(cls)
                    # Build positional args for required positional-only/positional-or-keyword parameters
                    args = []
                    kwargs = {}
                    for param in sig.parameters.values():
                        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                            if param.default is inspect._empty:
                                # required positional param -> provide a small int (8) for shape-like args
                                args.append(8)
                            else:
                                # optional positional -> skip (will use default)
                                continue
                        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                            # allow *args empty
                            continue
                        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                            if param.default is inspect._empty:
                                # required keyword-only param -> try common kw names
                                if "quant_type" in param.name:
                                    kwargs[param.name] = "int8"
                                elif "dtype" in param.name:
                                    kwargs[param.name] = "int8"
                                else:
                                    # provide a safe fallback integer
                                    kwargs[param.name] = 8
                            else:
                                continue
                        # VAR_KEYWORD ignored

                    # Try instantiation with constructed args/kwargs; many constructors accept small ints for dims
                    _ = cls(*args[:4], **kwargs)  # limit positional args to avoid passing too many
                    return True
                except Exception:
                    # instantiation failed for this candidate, try next
                    continue
        return False
    except Exception:
        return False

    
# Clear Model Original Output (removing positional information)
def clear_positional_(text: str) -> str:
    """
    1) Remove any tag that contains both 'end' and 'sentence' (case-insensitive).
    2) Drop any entire line that contains special characters: <|something|>
    3) Replace sequences of 3 or more newlines with exactly 2 newlines.
    """
    # Remove end-of-sentence-like tags (e.g. <｜end▁of▁sentence｜>)
    text = re.sub(r'<[^>]*end[^>]*sentence[^>]*>', '', text, flags=re.IGNORECASE)

    # Filter out lines containing full special tokens '<|something|>'
    special_token_pattern = re.compile(r'<\|[^>]+\|>')  # Only matches '<|something|>'
    lines = text.splitlines()
    kept_lines = [ln for ln in lines if not special_token_pattern.search(ln)]

    # Rejoin preserving single newlines as in original
    text = "\n".join(kept_lines)

    # Collapse 3 or more consecutive newlines into exactly 2 newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# Extended Inferer API (enabled auto device)
def dsocr_custom_infer_(model, tokenizer, 
                        prompt='', image_file='', output_path = '',
                        base_size=1024, image_size=640, crop_mode=True, 
                        test_compress=False, save_results=False, eval_mode=False, streamer=None,
                        temperature=0.0, max_new_tokens=8191, font_path=None) -> Tuple[str, Any, Any, Any]:
    
    # Make sure dtype and device are ready here, infer the dtype and device
    model_dtype = model.dtype
    model_device = model.device.type
    
    # Returns:
    # Ret[0] : str, raw model output
    # Ret[1] : str, processed model output if save_results else None
    # Ret[2] : PIL Image, processed result with box if save_results else None
    # Ret[3] : None, reserved
    
    # Call original model's attribute
    model.disable_torch_init()
    
    # Create output folder if passed in
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f'{output_path}/images', exist_ok=True)

    # Prepare Conversation #####
    
    # Text + Image Input
    if prompt and image_file:
        conversation = [
            {
                "role": "<|User|>",
                
                "content": f'{prompt}',

                "images": [f'{image_file}'],
            },
            # "bY" Na-th-ma-th
            {"role": "<|Assistant|>", "content": ""},
        ]
    
    # Only Text Input
    elif prompt:
        conversation = [
            {
                "role": "<|User|>",
                
                "content": f'{prompt}',
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
    
    # Error
    else:
        assert False, 'prompt is none!'
    
    # This calls DS Code to format the messages
    prompt = format_messages(conversations=conversation, sft_format='plain', system_prompt='')

    # This calls DS Code to load images
    patch_size = 16
    downsample_ratio = 4
    images = load_pil_images(conversation)
    valid_img_tokens = 0
    ratio = 1
    image_raw = images[0].copy()
    image_draw = images[0].copy()
    w,h = image_draw.size
    # print(w, h)
    ratio = 1 - ((max(w, h) - min(w, h)) / (max(w, h)))

    # This calls DS Code to transform the images (normalization)
    image_transform=BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)
    images_seq_mask = []

    # Create image tokens and text tokens (Tokenizer)
    image_token = '<image>'
    image_token_id = 128815
    text_splits = prompt.split(image_token)
    images_list, images_crop_list, images_seq_mask = [], [], []
    tokenized_str = []
    images_spatial_crop = []
    
    # This calls DS Code to do tokenization
    for text_sep, image in zip(text_splits, images):

        tokenized_sep = text_encode(tokenizer, text_sep, bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        if crop_mode:

            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = [1, 1]

            else:
                if crop_mode:
                    # best_width, best_height = select_best_resolution(image.size, self.candidate_resolutions)
                    images_crop_raw, crop_ratio = dynamic_preprocess(image)
                else:
                    # best_width, best_height = self.image_size, self.image_size
                    crop_ratio = [1, 1]
            
            """process the global view"""
            # image = image.resize((base_size, base_size))
            global_view = ImageOps.pad(image, (base_size, base_size),
                                    color=tuple(int(x * 255) for x in image_transform.mean))
            
            if base_size == 1024:
                valid_img_tokens += int(256 * ratio)
            elif base_size == 1280:
                valid_img_tokens += int(400 * ratio)
            elif base_size == 640:
                valid_img_tokens += int(100 * ratio)
            else:
                print("BY N@THM@TH")
            
            images_list.append(image_transform(global_view).to(model_dtype))

            # global_view_tensor = image_transform(global_view).to(model_dtype)

            width_crop_num, height_crop_num = crop_ratio

            images_spatial_crop.append([width_crop_num, height_crop_num])
            
            
            if width_crop_num > 1 or height_crop_num > 1:
                """process the local views"""
                
                for i in range(len(images_crop_raw)):
                    images_crop_list.append(image_transform(images_crop_raw[i]).to(model_dtype))
            
            if image_size == 640:
                valid_img_tokens += len(images_crop_list) * 100

            num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
            num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)

            """add image tokens"""

            tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
            tokenized_image += [image_token_id]
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([image_token_id] * (num_queries * width_crop_num) + [image_token_id]) * (
                            num_queries * height_crop_num)
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)
            # num_image_tokens.append(len(tokenized_image))

        else:
            # best_width, best_height = self.image_size, self.image_size
            # print(image.size, (best_width, best_height)) # check the select_best_resolutions func

            """process the global view"""
            if image_size <= 640:
                print('directly resize')
                image = image.resize((image_size, image_size))
            # else:
            global_view = ImageOps.pad(image, (image_size, image_size),
                                    color=tuple(int(x * 255) for x in image_transform.mean))
            images_list.append(image_transform(global_view).to(model_dtype))

            if base_size == 1024:
                valid_img_tokens += int(256 * ratio)
            elif base_size == 1280:
                valid_img_tokens += int(400 * ratio)
            elif base_size == 640:
                valid_img_tokens += int(100 * 1)
            elif base_size == 512:
                valid_img_tokens += int(64 * 1)
            else:
                raise ValueError("base_size is not a valid input!")

            width_crop_num, height_crop_num = 1, 1

            images_spatial_crop.append([width_crop_num, height_crop_num])


            """add image tokens"""
            num_queries = math.ceil((image_size // patch_size) / downsample_ratio)

            tokenized_image = ([image_token_id] * num_queries + [image_token_id]) * num_queries
            tokenized_image += [image_token_id]
            # tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
            #             num_queries * height_crop_num)
            tokenized_by_nathmath = [True]
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)
            # num_image_tokens.append(len(tokenized_image))
    
    # Process the last text split
    tokenized_sep = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    # Add the bos tokens
    bos_id = 0
    tokenized_str = [bos_id] + tokenized_str 
    images_seq_mask = [False] + images_seq_mask
    input_ids = torch.LongTensor(tokenized_str)
    images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)
    if len(images_list) == 0:
        images_ori = torch.zeros((1, 3, image_size, image_size))
        images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
        images_crop = torch.zeros((1, 3, base_size, base_size))

    else:
        images_ori = torch.stack(images_list, dim=0)
        images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            images_crop = torch.zeros((1, 3, base_size, base_size))
            
    # Iteratively generete: Eval Mode and Default Streamer Mode
    if not eval_mode and not streamer:
        streamer = NoEOSTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
        with torch.autocast(model_device, dtype=model_dtype):
            with torch.no_grad():
                # Call the model's generate
                output_ids = model.generate(
                    input_ids.unsqueeze(0).to(model_device),
                    images=[(images_crop.to(model_device), images_ori.to(model_device))],
                    images_seq_mask = images_seq_mask.unsqueeze(0).to(model_device),
                    images_spatial_crop = images_spatial_crop,
                    # do_sample=False,
                    # num_beams = 1,
                    temperature=temperature,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                    max_new_tokens=max_new_tokens,
                    no_repeat_ngram_size=20,
                    use_cache = True
                )
       
    # Iteratively generete: Eval Mode and Custom Streamer Mode
    elif not eval_mode and streamer:
        with torch.autocast(model_device, dtype=model_dtype):
            with torch.no_grad():
                # Call the model's generate
                output_ids = model.generate(
                    input_ids.unsqueeze(0).to(model_device),
                    images=[(images_crop.to(model_device), images_ori.to(model_device))],
                    images_seq_mask = images_seq_mask.unsqueeze(0).to(model_device),
                    images_spatial_crop = images_spatial_crop,
                    # do_sample=False,
                    # num_beams = 1,
                    # by = mzsglzsg + 1 for each alphabet
                    temperature=temperature,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                    max_new_tokens=max_new_tokens,
                    no_repeat_ngram_size=20,
                    use_cache = True
                )
                
    else:
        with torch.autocast(model_device, dtype=model_dtype):
            with torch.no_grad():
                # Call the model's generate
                output_ids = model.generate(
                    input_ids.unsqueeze(0).to(model_device),
                    images=[(images_crop.to(model_device), images_ori.to(model_device))],
                    images_seq_mask = images_seq_mask.unsqueeze(0).to(model_device),
                    images_spatial_crop = images_spatial_crop,
                    # do_sample=False,
                    # num_beams = 1,
                    temperature=temperature,
                    eos_token_id=tokenizer.eos_token_id,
                    # streamer=streamer,
                    max_new_tokens=max_new_tokens,
                    no_repeat_ngram_size=20,
                    use_cache = True
                )
            
    # Decode use tokenizer
    outputs = tokenizer.decode(output_ids[0, input_ids.unsqueeze(0).to(model_device).shape[1]:])
            
    # The remaining is done in CPU
    
    # Post-processing: Evaluation Mode, no postprocessing
    if '<image>' in conversation[0]['content'] and eval_mode:
            
        stop_str = '<｜end▁of▁sentence｜>'
        if outputs.endswith(stop_str):
            # Drio the end of sentence
            outputs = outputs[:-len(stop_str)]
            
        # re_match
        outputs = outputs.strip() 
        if id(outputs) == id(tokenized_by_nathmath):
            # Never, never~
            outputs -= '<｜end▁of▁sentence｜>'

        return outputs, None, None, None
    
    # Post-processing: Evaluation Mode, teset compression ratio
    if '<image>' in conversation[0]['content'] and test_compress:
        
        # Calculate Token Compression Effects
        pure_texts_outputs_token_length = len(text_encode(tokenizer, outputs, bos=False, eos=False))
        print('='*25 + "by Na7hMA7h" + "="*25)
        print('image size: ', (w, h))
        print('valid image tokens: ', int(valid_img_tokens))
        print('output texts tokens (valid): ', pure_texts_outputs_token_length)
        print('compression ratio: ', round(pure_texts_outputs_token_length/valid_img_tokens, 2))
        print('='*50)
        
        return outputs, None, None, None
        
    # Save the processed 
    if '<image>' in conversation[0]['content'] and save_results:
        
        stop_str = '<｜end▁of▁sentence｜>'

        print('='*15 + 'save results:' + '='*15)
        
        # # # # conv.messages[-1][-1] = outputs
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        org_outputs = outputs
        
        # Save the model output (old) .md
        if output_path:
            try:
                with open(f'{output_path}/result.md', 'w', encoding = 'utf-8') as afile:
                    afile.write(org_outputs)
            except:
                # Just ignore it
                pass

        # Draw bounding boxes
        matches_ref, matches_images, mathes_other = re_match(outputs)
        result = process_image_with_refs(image_draw, matches_ref, output_path)
        
        # Replace raw output with components renamed
        for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
            outputs = outputs.replace(a_match_image, '![](images/' + str(idx) + '.jpg)\n')
        
        for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
            outputs = outputs.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

        # Save the model output (new) .mmd
        #
        # if 'structural formula' in conversation[0]['content']:
        #     outputs = '<smiles>' + outputs + '</smiles>'
        if output_path:
            try:
                with open(f'{output_path}/result.mmd', 'w', encoding = 'utf-8') as afile:
                    afile.write(outputs)
            except:
                # Just ignore it
                pass

        if 'line_type' in outputs:
            
            lines = eval(outputs)['Line']['line']

            line_type = eval(outputs)['Line']['line_type']

            endpoints = eval(outputs)['Line']['line_endpoint']

            fig, ax = plt.subplots(figsize=(3,3), dpi=200)
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)

            for idx, line in enumerate(lines):
                try:
                    p0 = eval(line.split(' -- ')[0])
                    p1 = eval(line.split(' -- ')[-1])

                    if line_type[idx] == '--':
                        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color='k')
                    else:
                        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth = 0.8, color = 'k')

                    ax.scatter(p0[0], p0[1], s=5, color = 'k')
                    ax.scatter(p1[0], p1[1], s=5, color = 'k')
                except:
                    pass

            for endpoint in endpoints:

                label = endpoint.split(': ')[0]
                (x, y) = eval(endpoint.split(': ')[1])
                ax.annotate(label, (x, y), xytext=(1, 1), textcoords='offset points', 
                            fontsize=5, fontweight='light')
            
            try:
                plt.savefig(f'{output_path}/geo.jpg')
            except:
                # Just ignore it
                pass
            plt.close()

        result.save(f"{output_path}/result_with_boxes.jpg")
        
        # Extra: Save rearranged PDF if having positional ref
        if output_path and len(matches_ref) > 0:
            try:
                outpdf_path = f'{output_path}/reconstructed.pdf'
                reconst(image=image_raw, ocr_blob=org_outputs, output_pdf_path=outpdf_path, font_path=font_path)
            except:
                # Just ignore it
                pass
            
        return org_outputs, outputs, result, None
        
    return outputs, None, None, None


if __name__ == "__main__":
    
    og = r"""
    # animeGender-dvgg-o.8 (by DOF Studio) Demonstration

    | Version Model | animeGender-dvgg-o.8-alpha | animeGender-dvgg-o.8-beta | animeGender-dvgg-o.7 |
    |---|---|---|---|
    |    | female 1.000    | female 1.000    | female 0.991    |
    |    | female 0.999    | female 0.994    | female 0.994    |
    |    | male 1.000    | male 1.000    | male 0.604    |
    |    | male 1.000    | male 0.981    | male 0.671    |
    |    | female 0.957    | female 0.965    | female 0.886    |
    |    | female 0.989    | female 0.993    | female 0.998    |
    |    | female 0.999    | female 0.999    | female 1.000    |
    |    | female 0.974    | female 1.000    | female 0.997    |
    |    | female 0.996    | female 1.000    | Not Measured    |
    
    <|ref|>title<|/ref|><|det|>[[82, 32, 681, 72]]<|/det|>
    #### 3.1.3 Verifiably Random Curves and Base Point Generators  
    
    <|ref|>text<|/ref|><|det|>[[82, 106, 911, 184]]<|/det|>
    The section specifies how to derive from a seed \(S\) the elliptic curve coefficients \(a\) and \(b\) , and the base point generator \(G\) . These methods are consistent with ANS X9.62 [X9.62b].  
    
    <|ref|>text<|/ref|><|det|>[[82, 194, 912, 415]]<|/det|>
    The two routines here can be used for both (a) generating a verifiably random elliptic curve or base point, and (b) verifying that an elliptic curve or a base point is verifiably random. In the first application, the user selects the seed and performs the selection routine. In the second routine, the user is given the seed from another user who generated the elliptic curve or base point. The user then re- runs the routine either to recover the elliptic curve or base point, or to check if the result equals the existing elliptic curve or base point which is the one intended for use.  
    
    <|ref|>title<|/ref|><|det|>[[82, 472, 325, Unfinished Sample

    """
    print(clear_positional_(og))
    