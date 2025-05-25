python main.py \
    --layout_yolo_ckpt /workspace/paper_translation/doclayout_yolo_weight/doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt \
    --llm_ckpt_path workspace/paper_translation/save_model/checkpoint-34951 \
    --pdf_file_path /workspace/paper_translation/pdf/Image Captioning through Image Transformer.pdf \
    --output_pdf_file_path /workspace/paper_translation/output_image_captioning_KOR_fitted.pdf \
    --nanum_font_path ./font/NanumGothicBold.ttf