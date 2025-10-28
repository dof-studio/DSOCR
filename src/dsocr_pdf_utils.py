# Project DSOCR
#
# dsocr_pdf_utils.py
# Providing PDF I/O for DS OCR
# by dof-studio/Nathmath
# Open Source Under Apache 2.0 License
# Website: https://github.com/dof-studio/DSOCR

import io
import re
import os
import fitz # pip install pymupdf
import img2pdf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Any
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Color Universe
class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m' 

# Convert PDF to Images
def pdf2img(pdf_path, dpi=300, image_format="PNG", save_at=None) -> list:
    """
    Convert PDF to Images
    """
    if os.path.exists(pdf_path) == False:
        raise ValueError(f"Path pdf_path = {pdf_path} does not exist")
    
    images = []
    
    pdf_document = fitz.open(pdf_path)
    
    zoom = dpi / 150.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]

        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None

        if image_format.upper() == "PNG":
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
        else:
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
        
        images.append(img)
    
    pdf_document.close()
    
    if save_at is not None and os.path.exists(save_at):
        for i, img in enumerate(images):
            filename = os.path.join(save_at, f"image_{i}.{image_format.lower()}")
            img.save(filename)
    
    return images

# Convert images to a PDF
def img2pdf(pil_images: list, output_path:str):
    """
    Merge Images to a PDF
    """
    if not pil_images:
        return
    
    image_bytes_list = []
    
    for img in pil_images:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=95)
        img_bytes = img_buffer.getvalue()
        image_bytes_list.append(img_bytes)
    
    try:
        pdf_bytes = img2pdf.convert(image_bytes_list)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    except Exception as e:
        print(f"error: {e}")

    return

def _extract_boxes_from_det(det_str: str) -> List[Tuple[int,int,int,int]]:
    """
    Extract integer coordinates from a det string like "[[x1,y1,x2,y2],[...]]".
    Return list of (x1,y1,x2,y2). If the content is malformed (not multiples of 4 ints),
    return an empty list.
    """
    nums = re.findall(r'-?\d+', det_str)
    if not nums or len(nums) < 4:
        return []
    ints = [int(n) for n in nums]
    if len(ints) % 4 != 0:
        # drop trailing incomplete group(s)
        ints = ints[: (len(ints) // 4) * 4]
    boxes = []
    for i in range(0, len(ints), 4):
        boxes.append((ints[i], ints[i+1], ints[i+2], ints[i+3]))
    return boxes

def _parse_ocr_style(text: str) -> List[Dict[str, Any]]:
    """
    Parse the OCR-like blob. For each <|ref|> ... <|/ref|> followed by <|det|>...<|/det|>,
    capture label type, boxes, and the free text that follows until next <|ref|> or end.
    Skip entries where det is missing or corrupted.
    Return list of dicts: {'type': str, 'boxes': [(x1,y1,x2,y2),...], 'text': str}
    """
    results = []
    # find all ref blocks with their spans
    ref_iter = list(re.finditer(r'<\|ref\|>(.*?)<\|/ref\|>', text, flags=re.DOTALL))
    text_len = len(text)
    for idx, m in enumerate(ref_iter):
        label_type = m.group(1).strip()
        # search for the next <|det|> after this ref close
        start_search = m.end()
        det_open = text.find('<|det|>', start_search)
        if det_open == -1:
            # corrupted or missing det; skip
            continue
        det_close = text.find('<|/det|>', det_open)
        if det_close == -1:
            # corrupted det block; skip
            continue
        det_content = text[det_open + len('<|det|>'):det_close]
        boxes = _extract_boxes_from_det(det_content)
        if not boxes:
            # corrupted coordinates or none; skip
            continue
        # capture following free text until next <|ref|> occurrence
        next_ref_pos = text_len
        if idx + 1 < len(ref_iter):
            next_ref_pos = ref_iter[idx + 1].start()
        following_text = text[det_close + len('<|/det|>'): next_ref_pos].strip()
        # normalize whitespace
        following_text = re.sub(r'\s+\n', '\n', following_text)
        following_text = following_text.strip()
        results.append({'type': label_type, 'boxes': boxes, 'text': following_text})
    return results

def _load_font_prefer_dejavu(size: int):
    """
    Try to load DejaVuSans (better unicode coverage). Fall back to default.
    """
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

def _wrap_text_to_box(draw: ImageDraw.Draw, text: str, font: ImageFont.ImageFont, box_w: int) -> List[str]:
    """
    Wrap unicode text to fit into box_w width in pixels.
    Returns list of lines.
    """

    # If font.getsize is not fully reliable for some fonts, we iterate greedily
    words = text.split()
    if not words:
        return []
    lines = []
    cur = words[0]
    for w in words[1:]:
        test = cur + ' ' + w
        w_px = draw.textbbox((0,0), test, font=font)[2]
        if w_px <= box_w:
            cur = test
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    # If any line still too long (e.g. no spaces), force character wrap:
    final_lines = []
    for line in lines:
        if draw.textbbox((0,0), line, font=font)[2] <= box_w:
            final_lines.append(line)
        else:
            # character-level wrap
            acc = ""
            for ch in line:
                test = acc + ch
                if draw.textbbox((0,0), test, font=font)[2] <= box_w:
                    acc = test
                else:
                    final_lines.append(acc)
                    acc = ch
            if acc:
                final_lines.append(acc)
    return final_lines

# Convert OCR result with original image to an editable pdf
def reconstruct_pdf_from_ocr_blob(image: Image.Image, ocr_blob: str, output_pdf_path: str,
                                  use_scale_denominator: int = 999,
                                  title_font_size: int = 20,
                                  sub_title_font_size: int = 14,
                                  text_font_size: int = 14,
                                  font_path: str = None):
    """
    Replacement that writes non-image elements as selectable PDF text using reportlab.
    image: PIL.Image for cropping image regions.
    font_path: optional path to a .ttf or .otf font file to embed for all text. If None or fails,
               falls back to a standard PDF font (may lack full unicode).
    """
    entries = _parse_ocr_style(ocr_blob)
    if not entries:
        raise ValueError("No valid entries parsed from OCR blob.")

    page_w, page_h = image.size  # use image pixels as PDF points (1:1)
    # ensure output dir exists
    outdir = os.path.dirname(output_pdf_path)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    c = rl_canvas.Canvas(output_pdf_path, pagesize=(page_w, page_h))

    # register custom font if provided
    font_name = None
    if font_path:
        try:
            font_name = "CustomFontEmbedded"
            pdfmetrics.registerFont(TTFont(font_name, font_path))
        except Exception as e:
            try:
                print(f"Warning: failed to register font '{font_path}': {e}. Falling back to default PDF font.")
            except Exception:
                pass
            font_name = None

    # fallback font name
    if not font_name:
        # use Helvetica as a generic fallback (may not support unicode)
        font_name = "Helvetica"

    # helper: wrap text to fit width using pdfmetrics.stringWidth
    def _wrap_text_for_pdf(text: str, font: str, size: int, max_width: float) -> List[str]:
        # naive word-based wrap; falls back to char-wrap if needed
        words = text.split()
        if not words:
            return []
        lines = []
        cur = words[0]
        for w in words[1:]:
            test = cur + ' ' + w
            if pdfmetrics.stringWidth(test, font, size) <= max_width:
                cur = test
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        # ensure no line exceeds width (char wrap)
        final_lines = []
        for ln in lines:
            if pdfmetrics.stringWidth(ln, font, size) <= max_width:
                final_lines.append(ln)
            else:
                acc = ""
                for ch in ln:
                    test = acc + ch
                    if pdfmetrics.stringWidth(test, font, size) <= max_width:
                        acc = test
                    else:
                        if acc:
                            final_lines.append(acc)
                        acc = ch
                if acc:
                    final_lines.append(acc)
        return final_lines

    # iterate entries and render
    for entry in entries:
        typ = entry['type']
        boxes = entry['boxes']
        content = entry['text'] or ""
        for (x1r, y1r, x2r, y2r) in boxes:
            # scale coords from 0..use_scale_denominator to page points
            try:
                x1 = int(x1r / use_scale_denominator * page_w)
                y1 = int(y1r / use_scale_denominator * page_h)
                x2 = int(x2r / use_scale_denominator * page_w)
                y2 = int(y2r / use_scale_denominator * page_h)
            except Exception:
                continue
            # clamp
            x1 = max(0, min(page_w-1, x1))
            x2 = max(0, min(page_w, x2))
            y1 = max(0, min(page_h-1, y1))
            y2 = max(0, min(page_h, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            box_w = x2 - x1
            box_h = y2 - y1

            if typ.lower() == 'image':
                # crop and draw image at (x1, page_h - y2)
                try:
                    cropped = image.crop((x1, y1, x2, y2))
                    img_reader = ImageReader(cropped)
                    # reportlab drawImage uses bottom-left origin
                    c.drawImage(img_reader, x1, page_h - y2, width=box_w, height=box_h, preserveAspectRatio=False, mask='auto')
                except Exception:
                    # skip bad image
                    continue
            else:
                # select font size
                if typ.lower() == 'title':
                    size = title_font_size
                elif typ.lower() == "sub_title" or typ.lower() == "subtitle":
                    size = sub_title_font_size
                else:
                    size = text_font_size

                c.setFont(font_name, size)

                # clean leading markdown "##" or "#"
                content_clean = re.sub(r'^\s*#\s*', '', content, flags=re.MULTILINE).strip()
                content_clean = re.sub(r'^\s*##\s*', '', content_clean, flags=re.MULTILINE).strip()
                # wrap to box width
                lines = _wrap_text_for_pdf(content_clean, font_name, size, box_w)
                
                # compute line height (approx)
                line_h = size * 1.2
                max_lines = int(box_h // line_h)
                if max_lines <= 1:
                    max_lines = 1
                lines = lines[:max_lines]

                # starting baseline Y for the first (top) line:
                # box top in reportlab coords is page_h - y1
                top_y = page_h - y1
                # put first baseline slightly below the top edge (small inset)
                inset = 2
                baseline_y = top_y - inset - (size * 0.0)  # baseline offset; adjust if needed

                # draw each line top -> down (subtracting line_h)
                y_cursor = baseline_y
                for ln in lines:
                    # ensure we don't write above box top or below box bottom
                    if y_cursor < (page_h - y2) + (size * 0.0):
                        # we've reached bottom, stop
                        break
                    c.drawString(x1, y_cursor, ln)
                    y_cursor -= line_h

    c.showPage()
    c.save()
    return output_pdf_path


# @TODO
# Support keyword <table>
# e.g.
# Note we can regard this as an HTML
"""
<|ref|>table<|/ref|><|det|>[[0, 0, 999, 1000]]<|/det|>

<table><tr><td colspan="4">animeGender-dvgg-o.8 (by DOF Studio) Demonstration</td></tr><tr><td>Version</td><td colspan="2">Proposal</td><td>Previous</td></tr><tr><td>Model</td><td>animeGender-dvgg-o.8-alpha</td><td>animeGender-dvgg-o.8-beta</td><td>animeGender-dvgg-o.7</td></tr><tr><td></td><td>female 1.000</td><td>female 1.000</td><td>female 0.991</td></tr><tr><td></td><td>female 0.999</td><td>female 0.994</td><td>female 0.994</td></tr><tr><td></td><td>male 1.000</td><td>male 1.000</td><td>male 0.604</td></tr><tr><td></td><td>male 1.000</td><td>male 0.981</td><td>male 0.671</td></tr><tr><td></td><td>female 0.957</td><td>female 0.965</td><td>female 0.886</td></tr><tr><td></td><td>female 0.989</td><td>female 0.993</td><td>female 0.998</td></tr><tr><td></td><td>female 0.999</td><td>female 0.999</td><td>female 1.000</td></tr><tr><td></td><td>female 0.974</td><td>female 1.000</td><td>female 0.997</td></tr><tr><td></td><td>female 0.996</td><td>female 1.000</td><td>Not Measured</td></tr></table>

"""

# Moreover
# @TODO
# Need support of parsing Math and Markdown
# e.g.
"""
<|ref|>title<|/ref|><|det|>[[82, 32, 681, 72]]<|/det|>
#### 3.1.3 Verifiably Random Curves and Base Point Generators  

<|ref|>text<|/ref|><|det|>[[82, 106, 911, 184]]<|/det|>
The section specifies how to derive from a seed \(S\) the elliptic curve coefficients \(a\) and \(b\) , and the base point generator \(G\) . These methods are consistent with ANS X9.62 [X9.62b].  

<|ref|>text<|/ref|><|det|>[[82, 194, 912, 415]]<|/det|>
The two routines here can be used for both (a) generating a verifiably random elliptic curve or base point, and (b) verifying that an elliptic curve or a base point is verifiably random. In the first application, the user selects the seed and performs the selection routine. In the second routine, the user is given the seed from another user who generated the elliptic curve or base point. The user then re- runs the routine either to recover the elliptic curve or base point, or to check if the result equals the existing elliptic curve or base point which is the one intended for use.  

<|ref|>title<|/ref|><|det|>[[82, 472, 325, 508]]<|/det|>
#### 3.1.3.1 Curve Selection  

<|ref|>text<|/ref|><|det|>[[82, 520, 911, 599]]<|/det|>
Input: A "seed" octet string \(S\) of length \(g / 8\) octets, field size \(q\) , hash function Hash of output length hashlen octets, and field element \(a \in \mathbb{F}_q\) .  

<|ref|>text<|/ref|><|det|>[[82, 603, 463, 642]]<|/det|>
Output: A field element \(b \in \mathbb{F}_q\) or "failure".  

<|ref|>text<|/ref|><|det|>[[82, 654, 461, 690]]<|/det|>
Actions: Generate the element \(b\) as follows:  

<|ref|>text<|/ref|><|det|>[[102, 737, 280, 781]]<|/det|>
1. Let \(m = \lfloor \log_2 q\rfloor\) .  

<|ref|>text<|/ref|><|det|>[[102, 803, 280, 840]]<|/det|>
2. Let \(t = 8\) hashlen.  

<|ref|>text<|/ref|><|det|>[[102, 867, 308, 910]]<|/det|>
3. Let \(s = \lfloor (m - 1) / t\rfloor\) .  

<|ref|>text<|/ref|><|det|>[[102, 933, 657, 975]]<|/det|>
4. Let \(k = m - st\) if \(q\) is even, and let \(k = m - st - 1\) if \(q\) is odd.

"""