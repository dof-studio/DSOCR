# Project DSOCR
#
# dsocr_postprocess.py
# Providing Post Processing of Original Model Outputs

import re 
from PIL import Image, ImageDraw


def draw_outlines(
             text_result, image, *, 
             outline_color: str = "red",
             outline_width: int = 3,
             **kwargs
             ):
    
    result_image_pil = None
    
    # Define the pattern to find all coordinates like [[280, 15, 696, 997]]
    pattern = re.compile(r"<\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>")
    matches = list(pattern.finditer(text_result)) # Use finditer to get all matches

    if matches:
        
        # Create a copy of the original image to draw on
        image_with_bboxes = image.copy()
        draw = ImageDraw.Draw(image_with_bboxes)
        w, h = image.size # Get original image dimensions

        for match_ in matches:
            # Extract coordinates as integers
            coords_norm = [int(c) for c in match_.groups()]
            x1_norm, y1_norm, x2_norm, y2_norm = coords_norm
            
            # Scale the normalized coordinates (from 1000x1000 space) to the image's actual size
            x1 = int(x1_norm / 1000 * w)
            y1 = int(y1_norm / 1000 * h)
            x2 = int(x2_norm / 1000 * w)
            y2 = int(y2_norm / 1000 * h)
            
            # Draw the rectangle with a outline, n pixels wide
            draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=outline_width)
        
        result_image_pil = image_with_bboxes
        
    else:
        # If no coordinates are found in the text, fall back to original image
        result_image_pil = image
        
    return text_result, result_image_pil


def drop_positional():
    
    # Drop positional information and get a cleared mkdown
    pass
