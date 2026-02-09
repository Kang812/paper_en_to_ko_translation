
import os
import sys
import torch
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from doclayout_yolo import YOLOv10
from pdf2image import convert_from_path
from unsloth import FastLanguageModel
import nltk
from ollama import chat
from weasyprint import HTML, CSS

# Import existing utils
# Assuming utils/text_translation.py is in the same directory or python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from text_translation import layout_detect, image_to_text, text_translation

def sort_layout_boxes(boxes, classes, page_width, page_height):
    """
    Sorts bounding boxes to follow the reading order (Top->Bottom, Left->Right for 2 columns).
    
    Args:
        boxes: np.array of [cx, cy, w, h]
        classes: np.array of class indices
        page_width: width of the page
        
    Returns:
        sorted_indices: list of indices in reading order
    """
    if len(boxes) == 0:
        return []

    # Convert to [x1, y1, x2, y2] for easier processing
    # boxes is [cx, cy, w, h]
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    # Store items as dictionaries
    items = []
    for i in range(len(boxes)):
        items.append({
            'index': i,
            'x1': x1[i], 'y1': y1[i], 'x2': x2[i], 'y2': y2[i],
            'cx': boxes[i][0], 'cy': boxes[i][1],
            'class': classes[i]
        })

    # 1. Identify "Full Width" items (Titles, Headers, Footers) that span across columns
    # Threshold: width > 60% of page width, OR specific classes like Title(10), Footer(4), Header(5)
    full_width_threshold = page_width * 0.6
    full_width_classes = [4, 5, 10] # Footer, Header, Title
    
    # Group items into vertical sections separated by full-width items
    # We will simply sort strictly by Y first to identify the "flow"
    # But for 2-column, we need to be careful.
    
    # Simple Heuristic Strategy for 2-column papers:
    # 1. Sort all items by Y (top to bottom) loosely.
    # 2. Iterate and split into "Row" groups based on Y overlaps? No, that's complex.
    # 
    # Better Strategy:
    # 1. Define a "Column Split" line (usually center of page).
    # 2. Divide items into: 
    #    - Spanning Items (cross the split line significantly)
    #    - Left Column Items
    #    - Right Column Items
    # 3. Sort Spanning Items by Y.
    # 4. "Insert" Left/Right sets between Spanning items? 
    #    Actually, most papers are: Title (Span) -> Body (2 Col) -> References (2 Col).
    #    Sometimes Figures span w/h.
    
    # Let's try a recursive Y-banding approach or just a simple robust 2-pass string sort.
    # Pass 1: Primary Key = Top Y coordinates (rounded to nearest 10-20px?) -> Issue with columns.
    
    # Implementation of "Manhattan Layout" sorting:
    sorted_items = []
    
    # Filter out items that are clearly headers/footers to process them separately? 
    # No, we want them in order.
    
    # Sort by Y-center first to roughly order them.
    # But wait, left col top text comes before right col top text, even if they have same Y.
    # And Left col BOTTOM text comes BEFORE Right col TOP text.
    
    # Let's designate left/right boundaries.
    col_split = page_width / 2
    
    # Classify each item
    for item in items:
        w = item['x2'] - item['x1']
        h = item['y2'] - item['y1']
        
        # Check if it spans across the center significantly
        # If x1 < split and x2 > split, and width is substantial
        if item['x1'] < col_split and item['x2'] > col_split and w > (page_width * 0.4):
            item['col'] = 'span'
        elif item['cx'] < col_split:
            item['col'] = 'left'
        else:
            item['col'] = 'right'
            
    # Now we need to process vertically. 
    # We can detect "Spanning Breaks".
    # Sort all spanning items by Y.
    spanners = [it for it in items if it['col'] == 'span']
    spanners.sort(key=lambda x: x['y1'])
    
    # Create Y-intervals defined by spanners: [0, span1.y1], [span1.y2, span2.y1], ...
    intervals = []
    current_y = 0
    
    for span in spanners:
        intervals.append({'y_start': current_y, 'y_end': span['y1'], 'type': 'content'})
        intervals.append({'y_start': span['y1'], 'y_end': span['y2'], 'type': 'span', 'item': span})
        current_y = span['y2']
    intervals.append({'y_start': current_y, 'y_end': page_height, 'type': 'content'})
    
    final_order_indices = []
    
    for interval in intervals:
        if interval['type'] == 'span':
            final_order_indices.append(interval['item']['index'])
        else:
            # Content interval: contains left and right columns
            # Find all items completely (or mostly) within this Y range
            # AND are not 'span' (already handled)
            chunk_items = []
            for item in items:
                if item['col'] == 'span': continue
                
                # Check intersection/inclusion with interval
                # Use item center or substantial overlap
                item_cy = (item['y1'] + item['y2']) / 2
                if interval['y_start'] <= item_cy < interval['y_end']:
                    chunk_items.append(item)
            
            # Sort chunk items:
            # Left column items first (sorted by Y), then Right column items (sorted by Y)
            lefts = [it for it in chunk_items if it['col'] == 'left']
            rights = [it for it in chunk_items if it['col'] == 'right']
            
            lefts.sort(key=lambda x: x['y1'])
            rights.sort(key=lambda x: x['y1'])
            
            for it in lefts: final_order_indices.append(it['index'])
            for it in rights: final_order_indices.append(it['index'])
            
    return final_order_indices

def get_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_font_base64(font_path):
    try:
        with open(font_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Error loading font: {e}")
        return None

def generate_html(content_list, output_path, font_path=None):
    font_face_css = ""
    font_family = "'Nanum Gothic', sans-serif"
    
    if font_path:
        font_b64 = get_font_base64(font_path)
        if font_b64:
            font_face_css = f"""
            @font-face {{
                font-family: 'NanumGothicLocal';
                src: url(data:font/truetype;charset=utf-8;base64,{font_b64}) format('truetype');
                font-weight: normal;
                font-style: normal;
            }}
            """
            font_family = "'NanumGothicLocal', 'Nanum Gothic', sans-serif"

    html = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <style>
            {font_face_css}
            
            body {{
                font-family: {font_family};
                line-height: 1.8;
                max-width: 900px;
                margin: 40px auto;
                padding: 20px;
                color: #2c3e50;
                background-color: #f9f9f9;
            }}
            .paper-container {{
                background-color: white;
                padding: 50px;
                box-shadow: 0 0 15px rgba(0,0,0,0.05);
                border-radius: 5px;
            }}
            h1 {{
                font-size: 28px;
                text-align: center;
                margin-bottom: 20px;
                color: #000;
                font-weight: 800;
            }}
            h2 {{
                font-size: 22px;
                margin-top: 30px;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
                color: #2c3e50;
                font-weight: 700;
            }}
            h3 {{
                font-size: 18px;
                margin-top: 20px;
                color: #34495e;
            }}
            p {{
                margin-bottom: 15px;
                text-align: justify;
                word-break: keep-all;
            }}
            .figure-container {{
                text-align: center;
                margin: 30px 0;
            }}
            .figure-img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .caption {{
                font-size: 14px;
                color: #7f8c8d;
                margin-top: 10px;
                font-style: italic;
                text-align: center;
            }}
            .footer {{
                font-size: 12px;
                color: #bdc3c7;
                text-align: center;
                margin-top: 50px;
                border-top: 1px solid #eee;
                padding-top: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="paper-container">
    """
    
    for item in content_list:
        tag = item['tag']
        text = item.get('text', '')
        image_b64 = item.get('image', None)
        
        if tag == 'h1':
            html += f"<h1>{text}</h1>\n"
        elif tag == 'h2':
            html += f"<h2>{text}</h2>\n"
        elif tag == 'h3':
            html += f"<h3>{text}</h3>\n"
        elif tag == 'img':
            html += f"""
            <div class="figure-container">
                <img src="data:image/png;base64,{image_b64}" class="figure-img" />
            </div>
            """
        elif tag == 'caption':
            html += f"<div class=\"caption\">{text}</div>\n"
        elif tag == 'footer':
            html += f"<div class=\"footer\">{text}</div>\n"
        else: # p
            if text.strip():
                html += f"<p>{text}</p>\n"

    html += """
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return html

def html_to_pdf(html_path, output_path):
    try:
        # Use WeasyPrint to generate PDF
        print(f"Generating PDF using WeasyPrint from {html_path}...")
        HTML(filename=html_path).write_pdf(output_path)
        print(f"PDF saved to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to generate PDF automatically with WeasyPrint: {e}")
        return False

def run_html_translation(layout_ckpt, llm_ckpt, pdf_path, output_html_path, output_pdf_path, font_path, ollama_mode=True):
    # 1. Load Models
    print("Loading Layout Model...")
    layout_model = YOLOv10(layout_ckpt)
    
    ts_model, tokenizer = None, None
    if not ollama_mode:
        print("Loading Translation Model...")
        ts_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=llm_ckpt,
            max_seq_length=2048,
            load_in_4bit=True
        )

    # 2. Convert PDF to Images
    print("Converting PDF...")
    images = convert_from_path(pdf_path, dpi=200)
    
    content_list = []
    
    # 3. Process each page
    for i, image in enumerate(tqdm(images, desc="Processing Pages")):
        page_w, page_h = image.size
        
        # Detect Layout
        names_map, pred_bbox, pred_cls = layout_detect(
            layout_model, image, 
            confidence_threshold=0.25, 
            nms_iou_thresh=0.4, 
            class_agnostic=True
        )
        
        if len(pred_bbox) == 0:
            continue
            
        # Sort Boxes (Reading Order)
        sorted_indices = sort_layout_boxes(pred_bbox, pred_cls, page_w, page_h)
        
        # Mapping DocLayNet classes to HTML tags
        # 0: Caption, 1: Footnote, 2: Formula, 3: List-item, 4: Page-footer, 5: Page-header
        # 6: Picture, 7: Section-header, 8: Table, 9: Text, 10: Title
        
        for idx in sorted_indices:
            bbox = pred_bbox[idx]
            cls_id = int(pred_cls[idx])
            
            # Extract Image/Text
            # If it's Picture(6), Table(8), Formula(2) -> Treat as Image
            if cls_id in [2, 6, 8]:
                # Crop and Save Image
                cx, cy, w, h = bbox
                xmin = int(cx - (w/2))
                ymin = int(cy - (h/2))
                xmax = int(cx + (w/2))
                ymax = int(cy + (h/2))
                
                # Boundary check
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(page_w, xmax), min(page_h, ymax)
                
                if xmax <= xmin or ymax <= ymin: continue
                
                crop = image.crop((xmin, ymin, xmax, ymax))
                if crop.size[0] == 0: continue
                
                img_b64 = get_image_base64(crop)
                content_list.append({'tag': 'img', 'image': img_b64})
                
            else:
                # Treat as Text
                text_segments, _ = image_to_text(image, bbox)
                if not text_segments: continue
                
                raw_text = ' '.join(text_segments)
                
                # Translate
                translated_text = ""
                if not ollama_mode:
                    # Not implemented fully here, assumes ollama mainly based on user context
                    pass 
                else:
                    if cls_id in [6, 8]: # Should be image, but if fell through
                        continue
                        
                    prompt = f"""
                    You are a technical translator. Translate this text to Korean naturally.
                    Preserve numbers, bullet points, and formatting.
                    Input: {raw_text}
                    """
                    try:
                        resp = chat(
                                    model='translategemma:12b', 
                                    messages=[{'role':'user', 'content': prompt}],
                                    options={'repeat_penalty': 1.5, 'top_p':0.9})
                        translated_text = resp.message.content.replace("*", "")
                    except Exception as e:
                        print(f"Trans Error: {e}")
                        translated_text = raw_text
                
                # Determine Tag
                tag = 'p'
                if cls_id == 10: tag = 'h1' # Title
                elif cls_id == 7: tag = 'h2' # Section Header
                elif cls_id == 5: tag = 'h3' # Page Header (maybe ignore?)
                elif cls_id == 0: tag = 'caption'
                elif cls_id == 4: tag = 'footer'
                
                # Clean up artifacts
                translated_text = translated_text.replace('\n', ' ')
                content_list.append({'tag': tag, 'text': translated_text})

    # 4. Generate HTML
    print("Generating HTML...")
    generate_html(content_list, output_html_path, font_path)
    print(f"HTML saved to {output_html_path}")
    
    # 5. Convert to PDF
    print("Converting to PDF...")
    html_to_pdf(output_html_path, output_pdf_path)

if __name__ == "__main__":
    # Settings
    layout_ckpt = '/workspace/paper_translation/doclayout_yolo_weight/doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt'
    llm_ckpt = '/workspace/paper_translation/save_model/checkpoint-34951'
    pdf_path = '/workspace/paper_translation/pdf/en/ACE-Step_1.5_Pushing_the_Boundaries_of_Open-Source_Music.pdf'
    output_html = '/workspace/paper_translation/pdf/ko/ACE-Step_1.5_Pushing_the_Boundaries_of_Open-Source_Music.html'
    output_pdf = '/workspace/paper_translation/pdf/ko/ACE-Step_1.5_Pushing_the_Boundaries_of_Open-Source_Music.pdf'
    nanum_font_path = '/workspace/paper_translation/font/NanumGothicBold.ttf'
    
    run_html_translation(layout_ckpt, llm_ckpt, pdf_path, output_html, output_pdf, nanum_font_path, ollama_mode=True)
