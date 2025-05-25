from utils.text_translation import paper_translation
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import argparse

def main(args):
    print(f"Using Layout YOLO checkpoint: {args.layout_yolo_ckpt}")
    print(f"Using LLM checkpoint (hardcoded in function): {args.llm_ckpt_path}") # 실제로는 함수 내 경로 사용
    print(f"Input PDF: {args.pdf_file_path}")
    print(f"Output PDF will be: {args.output_pdf_file_path}")
    print(f"Using Font: {args.nanum_font_path}")

    paper_translation(args.layout_yolo_ckpt, 
                      args.llm_ckpt_path, 
                      args.pdf_file_path,
                      output_pdf_path=args.output_pdf_file_path,
                      font_path=args.nanum_font_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='yolo_obb_Training')
    parser.add_argument('--layout_yolo_ckpt',
                        type = str,
                        default="/workspace/paper_translation/doclayout_yolo_weight/doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt")
    parser.add_argument('--llm_ckpt_path',
                        type = str,
                        default="workspace/paper_translation/save_model/checkpoint-34951")
    parser.add_argument('--pdf_file_path',
                        type = str,
                        default='/workspace/paper_translation/pdf/Image Captioning through Image Transformer.pdf')
    parser.add_argument('--output_pdf_file_path',
                        type = str,
                        default='/workspace/paper_translation/output_image_captioning_KOR_fitted.pdf')
    parser.add_argument('--nanum_font_path',
                        type = str,
                        default='./font/NanumGothicBold.ttf')
    
    args = parser.parse_args()
    main(args)
    
