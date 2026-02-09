python main.py \
    --layout_yolo_ckpt /workspace/paper_translation/doclayout_yolo_weight/doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt \
    --ollama_mode True \
    --llm_ckpt_path /workspace/paper_translation/save_model/checkpoint-34951 \
    --pdf_file_path /workspace/paper_translation/pdf/en/ACE-Step_1.5_Pushing_the_Boundaries_of_Open-Source_Music.pdf \
    --output_pdf_file_path /workspace/paper_translation/pdf/ko/ACE-Step_1.5_Pushing_the_Boundaries_of_Open-Source_Music_KOR.pdf \
    --nanum_font_path ./font/NanumGothicBold.ttf \
    --output_format html