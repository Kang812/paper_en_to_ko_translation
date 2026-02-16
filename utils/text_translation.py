import os
import torch
import numpy as np
import re
import pdfplumber
import collections
import nltk # sent_tokenize 사용을 위해 nltk 다운로드 확인 로직 추가
try:
    from utils.lang_config import LangConfig
except:
    try:
        from lang_config import LangConfig
    except:
        from .lang_config import LangConfig

from pdf2image import convert_from_path
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph, Frame, KeepInFrame
from nltk.tokenize import sent_tokenize
from torchvision.ops import nms # NMS를 위해 torchvision.ops 임포
from tqdm import tqdm
from reportlab.pdfgen import canvas
from doclayout_yolo import YOLOv10 # 실제 사용하신다면 이 라이브러리가 설치되어 있어야 합니다.
from PIL import Image
from ollama import chat


# nltk 'punkt' 리소스 확인 및 다운로드 (sent_tokenize 사용에 필요)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading nltk 'punkt' tokenizer...")
    nltk.download('punkt')

def get_font_size_from_pdf(pdf_path, page_num, bbox):
    """
    주어진 페이지와 바운딩 박스 내에서 가장 빈도가 높은 폰트 크기를 추출합니다.
    bbox: [x0, top, x1, bottom] (pdfplumber 좌표계 기준)
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num >= len(pdf.pages):
                return None
            page = pdf.pages[page_num]
            
            x0, top, x1, bottom = bbox
            font_sizes = []
            
            # 해당 영역 내의 글자들만 필터링
            # page.chars는 딕셔너리 리스트: 'x0', 'top', 'x1', 'bottom', 'size' 등 포함
            for char in page.chars:
                c_x0 = char['x0']
                c_top = char['top']
                c_x1 = char['x1']
                c_bottom = char['bottom']
                
                # 중심점이 박스 안에 있는지 확인
                c_cx = (c_x0 + c_x1) / 2
                c_cy = (c_top + c_bottom) / 2
                
                if x0 <= c_cx <= x1 and top <= c_cy <= bottom:
                    font_sizes.append(char['size'])
            
            if not font_sizes:
                return None
            
            # 가장 빈도가 높은 폰트 크기 반환 (정수로 반올림하여 빈도 계산)
            rounded_sizes = [round(s) for s in font_sizes]
            if not rounded_sizes:
                return None
                
            most_common = collections.Counter(rounded_sizes).most_common(1)[0][0]
            return most_common
            
    except Exception as e:
        print(f"Error extracting font size: {e}")
        return None

def layout_detect(model, image, confidence_threshold=0.2, nms_iou_thresh=0.5, class_agnostic=True):
    """
    레이아웃을 감지하고 NMS를 적용하여 중복 바운딩 박스를 제거합니다.
    """
    det_res = model.predict(
        image,
        imgsz=1024,
        conf=confidence_threshold, # 모델 predict 시 사용되는 confidence threshold
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    
    names_map = det_res[0].names # 클래스 이름 맵
    
    # Ultralytics YOLO 결과에서 boxes 객체 가져오기
    boxes_obj = det_res[0].boxes
    
    # .xyxy, .conf, .cls 속성이 있는지 확인하고 PyTorch Tensor로 가져오기
    if boxes_obj is None or \
       boxes_obj.xyxy is None or boxes_obj.conf is None or boxes_obj.cls is None or \
       boxes_obj.xyxy.numel() == 0:
        empty_np_array = np.array([])
        return names_map, empty_np_array, empty_np_array

    # CPU로 데이터 이동
    xyxy_boxes = boxes_obj.xyxy.cpu()   # 바운딩 박스 [N, 4] (xmin, ymin, xmax, ymax)
    pred_conf = boxes_obj.conf.cpu()    # 신뢰도 [N]
    pred_cls_raw = boxes_obj.cls.cpu()  # 클래스 ID [N]

    final_indices_to_keep = []
    
    if class_agnostic:
        # 클래스 구분 없이 전체 박스에 대해 NMS 수행
        keep_indices = nms(xyxy_boxes, pred_conf, nms_iou_thresh)
        final_indices_to_keep = keep_indices.tolist()
    else:
        # 기존 로직: 클래스별로 NMS 수행
        unique_classes = torch.unique(pred_cls_raw)
        for cls_id in unique_classes:
            class_mask = (pred_cls_raw == cls_id)
            class_boxes_xyxy = xyxy_boxes[class_mask]
            class_scores = pred_conf[class_mask]
            
            # torchvision.ops.nms 적용
            keep_for_class = nms(class_boxes_xyxy, class_scores, nms_iou_thresh)
            
            original_indices_for_class = torch.where(class_mask)[0]
            final_indices_to_keep.extend(original_indices_for_class[keep_for_class].tolist())

    if not final_indices_to_keep:
        empty_np_array = np.array([])
        return names_map, empty_np_array, empty_np_array

    # 최종 선택된 인덱스를 사용하여 박스, 신뢰도, 클래스 필터링
    final_indices_to_keep = sorted(list(set(final_indices_to_keep)))
    final_indices_to_keep_tensor = torch.tensor(final_indices_to_keep, dtype=torch.long)

    final_xyxy_boxes = xyxy_boxes[final_indices_to_keep_tensor]
    final_pred_conf = pred_conf[final_indices_to_keep_tensor]
    final_pred_cls = pred_cls_raw[final_indices_to_keep_tensor]

    # 최종 xyxy 박스를 xywh 형식으로 변환 (원래 코드와의 일관성)
    x_center = (final_xyxy_boxes[:, 0] + final_xyxy_boxes[:, 2]) / 2
    y_center = (final_xyxy_boxes[:, 1] + final_xyxy_boxes[:, 3]) / 2
    widths = final_xyxy_boxes[:, 2] - final_xyxy_boxes[:, 0]
    heights = final_xyxy_boxes[:, 3] - final_xyxy_boxes[:, 1]
    
    # NumPy 배열로 변환
    pred_bbox_np = torch.stack((x_center, y_center, widths, heights), dim=1).numpy()
    pred_cls_np = final_pred_cls.numpy()
    pred_conf_np = final_pred_conf.numpy()

    return names_map, pred_bbox_np, pred_cls_np

def ocr_image(crop_image):
    image_path = '/workspace/paper_translation/utils/crop_image/crop_image.png'
    crop_image.save(image_path)
    
    response = chat(
        model='glm-ocr:bf16',
        messages=[{'role': 'user', 'content': 'Text Recognition', 'images':[image_path]}],
    )
    
    os.remove(image_path)
    text = response.message.content

    return text

def image_to_text(image, bbox):
    cx, cy, w, h = bbox

    xmin = int(cx - (w/2))
    ymin = int(cy - (h/2))
    xmax = int(cx + (w/2))
    ymax = int(cy + (h/2))
    
    if not isinstance(image, Image.Image):
        pil_image = Image.fromarray(np.array(image))
    else:
        pil_image = image

    crop_img = pil_image.crop((xmin, ymin, xmax, ymax))

    if 0 in crop_img.size: 
        print(f"Warning: Crop size is zero for bbox {bbox}. Original image part: {xmin, ymin, xmax, ymax}")
        return [], [xmin, ymin, xmax, ymax]
        
    text = ocr_image(crop_img)
    text = re.sub(r'[\n\x0c]+', ' ', text).strip() 
    if not text: 
        return [], [xmin, ymin, xmax, ymax]
        
    sentences = sent_tokenize(text)
    return sentences, [xmin, ymin, xmax, ymax]

def fit_text_to_frame(text_content, width, height, canvas_obj, base_style, target_font_size=None, min_font_size=6, max_font_size=24):
    """
    주어진 텍스트를 프레임 크기에 맞게 폰트 크기를 조절하여 Paragraph 객체를 반환합니다.
    target_font_size가 주어지면 해당 크기 근처에서 맞추려고 시도합니다.
    """
    if target_font_size:
        # 목표 폰트부터 시작 (단, max_font_size를 넘지 않도록)
        start_font_size = min(target_font_size, max_font_size)
        start_font_size = max(start_font_size, min_font_size)
    else:
        start_font_size = max_font_size
        
    current_font_size = start_font_size
    
    while current_font_size >= min_font_size:
        style = ParagraphStyle(
            'FittedStyle',
            parent=base_style,
            fontSize=current_font_size,
            leading=current_font_size * 1.2, 
        )
        para = Paragraph(text_content, style)
        
        text_w, text_h = para.wrapOn(canvas_obj, width, height) 
        
        if text_h <= height and text_w <= width : 
            return para 
            
        current_font_size -= 0.5 
        
    style = ParagraphStyle(
        'FittedStyleMin',
        parent=base_style,
        fontSize=min_font_size,
        leading=min_font_size * 1.2,
    )
    return Paragraph(text_content, style)

def paper_translation(layout_yolo_ckpt,
                      pdf_path,
                      source_lang = "영어",
                      target_lang = "한국어",
                      output_pdf_path="/workspace/paper_translation/output_final.pdf", 
                      font_path='/workspace/paper_translation/font/NanumGothicBold.ttf'):
    
    print("Loading layout detection model...")
    layout_detect_model = YOLOv10(layout_yolo_ckpt)
    
    print(f"Converting PDF to images: {pdf_path}")
    
    lang_config = LangConfig()
    source_lang = lang_config.get_ko_to_en(source_lang)
    target_lang = lang_config.get_ko_to_en(target_lang)

    source_lang_code = lang_config.get_lang_code(source_lang)
    target_lang_code = lang_config.get_lang_code(target_lang)


    try:
        images = convert_from_path(pdf_path, dpi=200)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return

    if not images:
        print("No images converted from PDF.")
        return
        
    pdfmetrics.registerFont(TTFont('NanumGothic', font_path))

    initial_max_fontSize = 40  # Base size in points (will be scaled)
    min_font_size_for_fit = 8 # Base minimum size in points

    styles = getSampleStyleSheet()
    
    base_font_style = ParagraphStyle(
        'BaseFontStyle',
        parent=styles['Normal'],
        fontName='NanumGothic',
        leading=14, 
        firstLineIndent=0,
    )
    
    img_width, img_height = images[0].size
    c = canvas.Canvas(output_pdf_path, pagesize=(img_width, img_height))
    page_width, page_height = img_width, img_height

    print("Translation and PDF generation started...")
    
    # PDF 파일 열기 (pdfplumber 이용)
    pdf_obj = None
    try:
        pdf_obj = pdfplumber.open(pdf_path)
    except Exception as e:
        print(f"Warning: Could not open PDF with pdfplumber for font size extraction: {e}")
    
    for i, image in enumerate(tqdm(images, desc="Processing pages")): 
        
        w_ratio = 1.0 
        h_ratio = 1.0
        
        names_map, pred_bbox, pred_cls = layout_detect(
            layout_detect_model, 
            image, 
            confidence_threshold=0.25, 
            nms_iou_thresh=0.4,       
            class_agnostic=True        
        )

        try:
            full_page_img_reader = ImageReader(image)
            c.drawImage(full_page_img_reader, 0, 0, width=page_width, height=page_height)
        except Exception as e:
            print(f"Error drawing background image for page {i+1}: {e}")

        if pred_bbox.size == 0: 
            print(f"No layout objects detected on page {i+1}.")
            if i < len(images) - 1:
                 c.showPage()
            continue

        for j in range(len(pred_bbox)):
            bbox_xywh = pred_bbox[j]
            cls_id = int(pred_cls[j]) 
            
            text_processing_classes = [0, 1, 3, 4, 5, 7, 9, 10]
            
            if cls_id not in text_processing_classes:
                continue

            text_list, box_coords = image_to_text(image, bbox_xywh) 
            
            if not text_list:
                continue
            
            xmin_orig, ymin_orig, xmax_orig, ymax_orig = box_coords
            frame_width = (xmax_orig - xmin_orig) / w_ratio
            frame_height = (ymax_orig - ymin_orig) / h_ratio
            frame_x = xmin_orig / w_ratio
            frame_y = page_height - (ymax_orig / h_ratio)

            if frame_width <= 0 or frame_height <= 0:
                continue
            
            padding_x = 2
            padding_y = 2
            
            c.setFillColor("white")
            c.setStrokeColor("white")
            c.rect(frame_x - padding_x, frame_y - padding_y, 
                   frame_width + (padding_x*2), frame_height + (padding_y*2), 
                   fill=1, stroke=0)
            
            en = '.'.join(text_list)
            prompt = f"""
            You are a professional {source_lang} ({source_lang_code}) to {target_lang} ({target_lang_code}) translator. 
            Your goal is to accurately convey the meaning and nuances of the original {source_lang} text while 
            adhering to {target_lang} grammar, vocabulary, and cultural sensitivities.
            Produce only the {target_lang} translation, without any additional explanations or commentary. 
            Please translate the following {source_lang} text into {target_lang}:
            
            Input Text:
            {en}
            """
            
            try:
                response = chat(
                    model='translategemma:12b',
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'repeat_penalty': 1.5, 'top_p': 0.9} 
                )
                text_content = response.message.content
            except Exception as e_ollama:
                print(f"Ollama error: {e_ollama}")
                text_content = en 

            if not text_content.strip():
                continue

            # 4. 폰트 크기 추출 (pdfplumber 이용)
            extracted_font_size = None
            if pdf_obj:
                try:
                    if i < len(pdf_obj.pages):
                        pdf_page = pdf_obj.pages[i]
                        pdf_w = pdf_page.width
                        pdf_h = pdf_page.height
                        
                        scale_x = img_width / float(pdf_w)
                        scale_y = img_height / float(pdf_h)
                        
                        pdf_x0 = xmin_orig / scale_x
                        pdf_top = ymin_orig / scale_y
                        pdf_x1 = xmax_orig / scale_x
                        pdf_bottom = ymax_orig / scale_y
                        
                        target_bbox = (pdf_x0, pdf_top, pdf_x1, pdf_bottom)
                        
                        # bbox로 폰트 크기 추출
                        extracted_font_size = get_font_size_from_pdf(pdf_path, i, target_bbox)
                except Exception as e_size:
                    # 조용히 넘어감 (기본 폰트 동작)
                    pass

            # 5. 번역된 텍스트 스타일 설정
            # Alignment is now handled in the DPI scaling block below
            current_alignment = TA_JUSTIFY
            
            current_text_style = ParagraphStyle(
                f'TextStyle_Page{i}_Box{j}',
                parent=base_font_style,
                alignment=current_alignment,
            )
            
            target_min_font = min_font_size_for_fit
            target_max_font = initial_max_fontSize
            
            target_size = extracted_font_size if extracted_font_size else None
            
            # DPI Scaling Factor (PDF points to Image pixels)
            DPI_SCALE = 200 / 72.0 
            
            if cls_id == 10: 
                # Title: usually 24pt -> ~66px
                if target_size: target_size = max(target_size * DPI_SCALE * 1.2, 24 * DPI_SCALE) 
                else: target_size = 28 * DPI_SCALE
                target_min_font = 18 * DPI_SCALE
                current_text_style.alignment = TA_CENTER
                current_text_style.leading = target_size * 1.2
            elif cls_id == 7: 
                # Section Header: usually 12-14pt -> ~33-39px
                if target_size: target_size = max(target_size * DPI_SCALE * 1.1, 14 * DPI_SCALE)
                else: target_size = 14 * DPI_SCALE
                target_min_font = 11 * DPI_SCALE
                current_text_style.alignment = TA_JUSTIFY
                current_text_style.leading = target_size * 1.2
            elif cls_id == 9: # Text
                # Body Text: usually 10-11pt -> ~28-30px
                if target_size: target_size = max(target_size * DPI_SCALE * 1.0, 10 * DPI_SCALE)
                else: target_size = 10 * DPI_SCALE
                target_min_font = 9 * DPI_SCALE # Minimum readable size
                current_text_style.alignment = TA_JUSTIFY
                current_text_style.leading = target_size * 1.3 # Slightly more leading for Korean body
            else:
                 # Default logic for others
                 if target_size: target_size = max(target_size * DPI_SCALE, 10 * DPI_SCALE)
                 else: target_size = 10 * DPI_SCALE
                 target_min_font = 8 * DPI_SCALE

            para = fit_text_to_frame(text_content, frame_width, frame_height, c, current_text_style, 
                                     target_font_size=target_size, 
                                     min_font_size=target_min_font, 
                                     max_font_size=target_max_font * DPI_SCALE)
            
            text_frame = Frame(frame_x, frame_y, frame_width, frame_height, showBoundary=0,
                               leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
            
            kif = KeepInFrame(frame_width, frame_height, [para], mode='shrink') 
            text_frame.addFromList([kif], c)
        
        if i < len(images) - 1:
            c.showPage()
            
    if pdf_obj:
        pdf_obj.close()

    c.save()
    print(f"PDF saved to {output_pdf_path}")

if __name__ == '__main__':
    layout_yolo_ckpt = '/workspace/paper_translation/doclayout_yolo_weight/doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt' 
    
    pdf_file_path = '/workspace/paper_translation/pdf/en/2003.08934v2.pdf'
    output_pdf_file_path = '/workspace/paper_translation/pdf/ko/2003.08934v2_KOR.pdf'
    nanum_font_path = '/workspace/paper_translation/font/NanumGothicBold.ttf' 

    print(f"Using Layout YOLO checkpoint: {layout_yolo_ckpt}")
    print(f"Input PDF: {pdf_file_path}")
    print(f"Output PDF will be: {output_pdf_file_path}")
    
    try:
        import pdfplumber
    except ImportError:
        print("Warning: pdfplumber is not installed. Font size preservation will not work optimally.")
        print("Please install it: pip install pdfplumber")

    paper_translation(layout_yolo_ckpt, 
                      pdf_file_path,
                      source_lang = "영어",
                      target_lang = "한국어",
                      output_pdf_path=output_pdf_file_path,
                      font_path=nanum_font_path)
