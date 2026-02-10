from tqdm import tqdm
import torch
import numpy as np
# import base64 # base64는 코드에서 사용되지 않아 주석 처리
# import matplotlib.pyplot as plt # matplotlib.pyplot는 코드에서 사용되지 않아 주석 처리
# import io # io는 코드에서 사용되지 않아 주석 처리
from reportlab.pdfgen import canvas
from doclayout_yolo import YOLOv10 # 실제 사용하신다면 이 라이브러리가 설치되어 있어야 합니다.
from PIL import Image # ImageChops는 사용되지 않아 제거
# import matplotlib as mpl # matplotlib.mpl은 코드에서 사용되지 않아 주석 처리
import re
from unsloth import FastLanguageModel
from pdf2image import convert_from_path
import pytesseract
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph, Frame, KeepInFrame
from nltk.tokenize import sent_tokenize
import torch
from torchvision.ops import nms # NMS를 위해 torchvision.ops 임포
import nltk # sent_tokenize 사용을 위해 nltk 다운로드 확인 로직 추가
from ollama import chat

# nltk 'punkt' 리소스 확인 및 다운로드 (sent_tokenize 사용에 필요)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading nltk 'punkt' tokenizer...")
    nltk.download('punkt')

def translate_model(model, tokenizer, text, device):
    messages = [{"role": "user", "content": text}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=128, # 번역 결과 길이에 따라 조절 가능
        use_cache=True,
        # temperature=3.5, # 매우 높은 값, 좀 더 일반적인 값(예: 0.7-1.0) 또는 제거 고려
        # min_p=0.9          # nucleus sampling, top_p와 유사. 필요에 따라 조절
        # 더 일반적인 생성 파라미터 예시:
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    translation = tokenizer.batch_decode(outputs)[0]
    translation = translation.split("<|end_header_id|>")[-1] \
                             .replace("\n\n", " ") \
                             .replace("<|eot_id|>", "").strip() # 양 끝 공백 제거 추가
    return translation

def layout_detect(model, image, confidence_threshold=0.2, nms_iou_thresh=0.5, class_agnostic=True):
    """
    레이아웃을 감지하고 NMS를 적용하여 중복 바운딩 박스를 제거합니다.

    Args:
        model: YOLOv10 모델 객체.
        image: PIL Image 또는 NumPy 배열 형식의 입력 이미지.
        confidence_threshold (float): 객체 감지를 위한 최소 신뢰도.
        nms_iou_thresh (float): NMS를 위한 IoU 임계값. 이 값보다 IoU가 큰 중복 박스가 제거됨.
        class_agnostic (bool): True일 경우 클래스에 상관없이 NMS를 적용하여 겹치는 박스를 제거함. 
                               False일 경우 같은 클래스끼리만 NMS 적용.

    Returns:
        tuple: (names, pred_bbox, pred_cls, pred_conf)
            names (dict): 클래스 ID와 이름 매핑.
            pred_bbox (np.array): 필터링된 바운딩 박스 (xywh 형식).
            pred_cls (np.array): 필터링된 바운딩 박스의 클래스 ID.
            pred_conf (np.array): 필터링된 바운딩 박스의 신뢰도.
    """
    det_res = model.predict(
        image,
        imgsz=1024,
        conf=confidence_threshold, # 모델 predict 시 사용되는 confidence threshold
        # model.predict 내부에서도 NMS가 수행됩니다.
        # 여기서는 추가적인 NMS 또는 다른 IoU 임계값으로 NMS를 수행하려는 경우에 유용합니다.
        # 만약 model.predict에 iou 파라미터가 있다면 그것을 조절하는 것이 첫 번째 단계일 수 있습니다.
        # 예: iou=0.4 (내부 NMS의 IoU 임계값)
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    
    names_map = det_res[0].names # 클래스 이름 맵
    
    # Ultralytics YOLO 결과에서 boxes 객체 가져오기
    boxes_obj = det_res[0].boxes
    
    # .xyxy, .conf, .cls 속성이 있는지 확인하고 PyTorch Tensor로 가져오기
    # 이 속성들이 None이거나 비어있으면 빈 배열 반환
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
            
            # keep_for_class는 class_boxes_xyxy 내의 인덱스입니다.
            # 원래 전체 박스 리스트에서의 인덱스로 변환합니다.
            original_indices_for_class = torch.where(class_mask)[0]
            final_indices_to_keep.extend(original_indices_for_class[keep_for_class].tolist())

    if not final_indices_to_keep:
        empty_np_array = np.array([])
        return names_map, empty_np_array, empty_np_array

    # 최종 선택된 인덱스를 사용하여 박스, 신뢰도, 클래스 필터링
    # 중복된 인덱스가 있을 수 있으므로 unique 처리 후 정렬 (class_agnostic=True인 경우 nms가 이미 unique한 결과를 주지만, False인 경우 합칠 때 순서 보장을 위해)
    final_indices_to_keep = sorted(list(set(final_indices_to_keep)))
    final_indices_to_keep_tensor = torch.tensor(final_indices_to_keep, dtype=torch.long)

    final_xyxy_boxes = xyxy_boxes[final_indices_to_keep_tensor]
    final_pred_conf = pred_conf[final_indices_to_keep_tensor]
    final_pred_cls = pred_cls_raw[final_indices_to_keep_tensor]

    # 최종 xyxy 박스를 xywh 형식으로 변환 (원래 코드와의 일관성)
    # pred_bbox = [xc, yc, width, height]
    x_center = (final_xyxy_boxes[:, 0] + final_xyxy_boxes[:, 2]) / 2
    y_center = (final_xyxy_boxes[:, 1] + final_xyxy_boxes[:, 3]) / 2
    widths = final_xyxy_boxes[:, 2] - final_xyxy_boxes[:, 0]
    heights = final_xyxy_boxes[:, 3] - final_xyxy_boxes[:, 1]
    
    # NumPy 배열로 변환
    pred_bbox_np = torch.stack((x_center, y_center, widths, heights), dim=1).numpy()
    pred_cls_np = final_pred_cls.numpy()
    pred_conf_np = final_pred_conf.numpy()

    return names_map, pred_bbox_np, pred_cls_np

def image_to_text(image, bbox):
    cx, cy, w, h = bbox

    xmin = int(cx - (w/2))
    ymin = int(cy - (h/2))
    xmax = int(cx + (w/2))
    ymax = int(cy + (h/2))
    
    # PIL Image 객체로 변환 (만약 image가 numpy 배열이 아니라면)
    if not isinstance(image, Image.Image):
        pil_image = Image.fromarray(np.array(image))
    else:
        pil_image = image

    # np.array(image) 대신 pil_image 사용
    crop_img = pil_image.crop((xmin, ymin, xmax, ymax))

    if 0 in crop_img.size: # 크롭된 이미지 크기가 0인 경우 방지
        # 이 경우 원본에서 약간 다른 좌표로 시도하거나 빈 텍스트 반환
        print(f"Warning: Crop size is zero for bbox {bbox}. Original image part: {xmin, ymin, xmax, ymax}")
        return [], [xmin, ymin, xmax, ymax]
        
    text = pytesseract.image_to_string(crop_img, lang="eng")
    
    text = re.sub(r'[\n\x0c]+', ' ', text).strip() # 양 끝 공백 제거 추가
    if not text: # 추출된 텍스트가 없을 경우 빈 리스트 반환
        return [], [xmin, ymin, xmax, ymax]
        
    sentences = sent_tokenize(text)
    return sentences, [xmin, ymin, xmax, ymax]

def text_translation(text_list, ts_model, tokenizer):
    en_to_ko_list = []
    
    for en_text in text_list:
        if not en_text.strip(): # 빈 문자열은 번역 시도하지 않음
            continue
        if en_text.lower() == 'abstract': # 'abstract'는 특별 취급 가능 (번역 안 함)
            en_to_ko_list.append("요약") # 예시: 'Abstract'를 '요약'으로
        else:
            # 실제 번역 모델 호출 시, 예외 처리 추가 가능성
            try:
                translation_en_to_ko = translate_model(ts_model, tokenizer, en_text, 
                                                       torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                en_to_ko_list.append(translation_en_to_ko)
            except Exception as e:
                print(f"Error during translation of '{en_text}': {e}")
                en_to_ko_list.append(f"[번역 오류] {en_text}") # 오류 발생 시 원문과 함께 표시

    text_en_to_ko = ' '.join(en_to_ko_list)
    return text_en_to_ko

def fit_text_to_frame(text_content, width, height, canvas_obj, base_style, min_font_size=6, max_font_size=24):
    """
    주어진 텍스트를 프레임 크기에 맞게 폰트 크기를 조절하여 Paragraph 객체를 반환합니다.
    """
    current_font_size = max_font_size
    
    while current_font_size >= min_font_size:
        # 현재 폰트 크기로 스타일 업데이트 (새 ParagraphStyle 객체 생성 또는 복사)
        style = ParagraphStyle(
            'FittedStyle',
            parent=base_style,
            fontSize=current_font_size,
            leading=current_font_size * 1.2, # 줄 간격도 폰트 크기에 비례하여 조절
        )
        para = Paragraph(text_content, style)
        
        # Paragraph가 프레임 내에 맞는지 확인
        text_w, text_h = para.wrapOn(canvas_obj, width, height) # wrapOn은 사용 가능한 너비와 높이를 받음
        
        if text_h <= height and text_w <= width : # 높이와 너비 모두 만족해야 함
            return para # 적합한 Paragraph 반환
            
        current_font_size -= 1 # 폰트 크기를 1 줄이고 다시 시도
        
    # 최소 폰트 크기로도 맞지 않으면 최소 폰트 크기로 Paragraph 반환
    # (또는 다른 처리: 예외 발생, 빈 Paragraph 반환 등)
    style = ParagraphStyle(
        'FittedStyleMin',
        parent=base_style,
        fontSize=min_font_size,
        leading=min_font_size * 1.2,
    )
    # print(f"Warning: Text could not fit perfectly even at min_font_size {min_font_size}. Content starts with: {text_content[:50]}...")
    return Paragraph(text_content, style)

def paper_translation(layout_yolo_ckpt,
                      llm_ckpt, # 이 인자는 FastLanguageModel.from_pretrained에서 model_name으로 사용됩니다.
                      pdf_path,
                      ollama_mode=False,
                      output_pdf_path="/workspace/paper_translation/output_final.pdf", # 출력 파일명 변경
                      font_path='/workspace/paper_translation/font/NanumGothicBold.ttf'):
    
    print("Loading layout detection model...")
    layout_detect_model = YOLOv10(layout_yolo_ckpt)

    print("Loading translation LLM...")
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    if not ollama_mode:
        ts_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=llm_ckpt, # llm_ckpt 변수 사용
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

    print(f"Converting PDF to images: {pdf_path}")
    
    try:
        images = convert_from_path(pdf_path, dpi=200)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return

    if not images:
        print("No images converted from PDF.")
        return
        
    pdfmetrics.registerFont(TTFont('NanumGothic', font_path))


    
    # 폰트 크기 설정값 조정
    initial_max_fontSize = 24 
    min_font_size_for_fit = 8 # 최소 폰트 크기 약간 조정

    styles = getSampleStyleSheet()
    
    # 기본 폰트 스타일 정의 (나눔고딕)
    base_font_style = ParagraphStyle(
        'BaseFontStyle',
        parent=styles['Normal'],
        fontName='NanumGothic',
        leading=14, # 기본 행간
        firstLineIndent=0,
    )
    
    img_width, img_height = images[0].size
    c = canvas.Canvas(output_pdf_path, pagesize=(img_width, img_height))
    page_width, page_height = img_width, img_height

    print("Translation and PDF generation started...")
    
    for i, image in enumerate(tqdm(images, desc="Processing pages")): 
        
        w_ratio = 1.0 
        h_ratio = 1.0
        
        # layout_detect 호출
        names_map, pred_bbox, pred_cls = layout_detect(
            layout_detect_model, 
            image, 
            confidence_threshold=0.25, 
            nms_iou_thresh=0.4,       
            class_agnostic=True        
        )

        # 1. 페이지 전체 이미지를 배경으로 그리기
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
            
            # 카테고리 정의 (DocLayNet 기준)
            # 0: Caption, 1: Footnote, 2: Formula, 3: List-item, 4: Page-footer, 5: Page-header
            # 6: Picture, 7: Section-header, 8: Table, 9: Text, 10: Title
            
            text_processing_classes = [0, 1, 3, 4, 5, 7, 9, 10]
            
            # 클래스별 정렬 및 스타일 설정
            # Title(10): 중앙 정렬, 큰 폰트 선호
            # Section-header(7): 좌측 정렬, 볼드
            # Page-header(5), Page-footer(4): 중앙 또는 좌측, 작은 폰트
            # Cipher/Caption(0): 중앙 또는 좌측
            # List-item(3): 좌측
            
            if cls_id not in text_processing_classes:
                continue

            # === 텍스트 요소 처리 ===
            text_list, box_coords = image_to_text(image, bbox_xywh) 
            
            if not text_list:
                continue
            
            # PDF 프레임 좌표 및 크기
            xmin_orig, ymin_orig, xmax_orig, ymax_orig = box_coords
            frame_width = (xmax_orig - xmin_orig) / w_ratio
            frame_height = (ymax_orig - ymin_orig) / h_ratio
            frame_x = xmin_orig / w_ratio
            frame_y = page_height - (ymax_orig / h_ratio)

            if frame_width <= 0 or frame_height <= 0:
                continue
            
            # 2. 원본 텍스트 가리기 (흰색 사각형 + 패딩)
            # 패딩을 추가하여 원본 글자가 삐져나오는 것을 방지
            padding_x = 2
            padding_y = 2
            
            c.setFillColor("white")
            c.setStrokeColor("white")
            # x, y, width, height (y는 아래 기준)
            # frame_y는 아래쪽 좌표이므로, 사각형을 그릴 때는 약간 더 아래로 내려야 함(padding_y)
            c.rect(frame_x - padding_x, frame_y - padding_y, 
                   frame_width + (padding_x*2), frame_height + (padding_y*2), 
                   fill=1, stroke=0)
            
            # 3. 텍스트 번역
            if not ollama_mode:
                text_content = text_translation(text_list, ts_model, tokenizer)
            else:
                en = '.'.join(text_list)
                # 프롬프트 개선: 자연스러운 한국어 연구 논문 스타일 요청
                prompt = f"""
                Translate the following academic paper text from English to Korean.
                Context: This is part of a research paper (Computer Science/AI domain).
                Guidelines:
                1. Maintain a professional, academic tone (e.g., use '습니다/합니다' or concise endings like '함/임' where appropriate for headers).
                2. Preserve formatting such as bullet points (*), numbers (1.), or math variables if present and relevant.
                3. Do NOT add explanations. Output ONLY the translated text.
                
                Input Text:
                {en}
                """
                
                try:
                    response = chat(
                        model='translategemma:12b',
                        messages=[{'role': 'user', 'content': prompt}],
                        options={'repeat_penalty': 1.1, 'top_p': 0.9} # 반복 페널티 약간 완화
                    )
                    text_content = response.message.content
                except Exception as e_ollama:
                    print(f"Ollama error: {e_ollama}")
                    text_content = en 

            if not text_content.strip():
                continue

            # 4. 번역된 텍스트 스타일 설정 (클래스 기반)
            current_alignment = TA_JUSTIFY # 기본값: 양쪽 정렬
            
            if cls_id == 10: # Title
                current_alignment = TA_CENTER
            elif cls_id == 7: # Section Header
                current_alignment = TA_JUSTIFY # 혹은 TA_LEFT
                # 헤더는 보통 꽉 차지 않으므로 Left가 나을 수 있으나, Justify도 무방
            elif cls_id in [0, 4, 5]: # Caption, Footer, Header
                current_alignment = TA_CENTER
            
            current_text_style = ParagraphStyle(
                f'TextStyle_Page{i}_Box{j}',
                parent=base_font_style,
                alignment=current_alignment,
                # 한글 단어 단위 줄바꿈을 위해 wordWrap 설정 (ReportLab 기본 동작 확인 필요)
            )
            
            # Title(10)이나 Header(7)는 폰트를 좀 더 키워도 됨
            target_min_font = min_font_size_for_fit
            target_max_font = initial_max_fontSize
            
            if cls_id == 10: # Title
                target_min_font = 14
            elif cls_id == 7: # Section Header
                target_min_font = 10
            
            para = fit_text_to_frame(text_content, frame_width, frame_height, c, current_text_style, 
                                     min_font_size=target_min_font, 
                                     max_font_size=target_max_font)
            
            # Frame 생성 (패딩 제거 또는 최소화, 이미 박스로 가렸으므로)
            text_frame = Frame(frame_x, frame_y, frame_width, frame_height, showBoundary=0,
                               leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
            
            kif = KeepInFrame(frame_width, frame_height, [para], mode='shrink') # mode='shrink' or 'truncate'
            text_frame.addFromList([kif], c)
        
        if i < len(images) - 1:
            c.showPage()
            
    c.save()
    print(f"PDF saved to {output_pdf_path}")

if __name__ == '__main__':
    # 경로 설정 (사용자 환경에 맞게 수정)
    layout_yolo_ckpt = '/workspace/paper_translation/doclayout_yolo_weight/doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt' 
    llm_ckpt_path = '/workspace/paper_translation/save_model/checkpoint-34951' 
    
    # 테스트용 파일 경로 (존재 여부 확인 필요)
    pdf_file_path = '/workspace/paper_translation/pdf/en/2003.08934v2.pdf'
    output_pdf_file_path = '/workspace/paper_translation/pdf/ko/2003.08934v2_KOR.pdf'
    nanum_font_path = '/workspace/paper_translation/font/NanumGothicBold.ttf' 

    print(f"Using Layout YOLO checkpoint: {layout_yolo_ckpt}")
    print(f"Input PDF: {pdf_file_path}")
    print(f"Output PDF will be: {output_pdf_file_path}")

    paper_translation(layout_yolo_ckpt, 
                      llm_ckpt_path, 
                      pdf_file_path,
                      ollama_mode=True,
                      output_pdf_path=output_pdf_file_path,
                      font_path=nanum_font_path)
