
import os
import streamlit as st
import tempfile
import argparse
import shutil
from utils.translation import translate_text
from utils.html_translation import run_html_translation
from utils.text_translation import paper_translation
from utils.lang_config import LangConfig
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="논문 번역 앱",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    
    html, body, [class*="css"] {
        font-family: 'Pretendard', sans-serif;
    }
    
    .stApp {
        background-color: #f8f9fa;
        color: #000000;
    }
    
    /* Global Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #02ab21;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #028a1b;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Card Styling for st.container(border=True) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
        margin-bottom: 1.5rem;
    }

    h1 {
        color: #1a1a1a;
        font-weight: 700;
        margin-bottom: 1.5rem !important;
    }
    
    .stMarkdown p {
        color: #4a4a4a;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

def main():
    with st.sidebar:
        choice = option_menu("메뉴", ["번역기", "문서 번역"],
                             icons=['translate', 'file-text'],
                             menu_icon="cast", default_index=1,
                             styles={
                                 "container": {"padding": "5!important", "background-color": "#ffffff"},
                                 "icon": {"color": "#02ab21", "font-size": "20px"}, 
                                 "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px", "--hover-color": "#f0f2f6", "font-family": "Pretendard"},
                                 "nav-link-selected": {"background-color": "#e8f5e9", "color": "#02ab21", "font-weight": "600"},
                             }
                            )

    if choice == "번역기":
        st.title("🤖 AI 번역기")
        
        # Load Language Config
        lang_config = LangConfig()
        lang_options = list(lang_config.lang_ko_en.keys())
        
        # Initialize session state for translation result if not exists
        if 'translated_text' not in st.session_state:
            st.session_state.translated_text = ""
            
        with st.container(border=True):
            # Language Selection
            col1, col2 = st.columns(2)
            with col1:
                source_lang = st.selectbox("원문 언어 (Source)", options=lang_options, index=lang_options.index("영어"))
            with col2:
                target_lang = st.selectbox("번역 언어 (Target)", options=lang_options, index=lang_options.index("한국어"))
            
            st.markdown("---")
            
            # Text Areas
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 원문 입력")
                input_text = st.text_area("번역할 텍스트를 입력하세요", height=400, label_visibility="collapsed", placeholder="여기에 번역할 내용을 입력하세요...")
            
            with c2:
                st.markdown("### 번역 결과")
                st.text_area("번역 결과", value=st.session_state.translated_text, height=400, label_visibility="collapsed", disabled=False)

            # Check for changes in input (to clear previous result if needed, optional)
            
            # Translate Button
            if st.button("🚀 번역하기", use_container_width=True):
                if input_text:
                    with st.spinner("번역 중입니다..."):
                        try:
                            translated_result = translate_text(input_text, source_lang, target_lang)
                            st.session_state.translated_text = translated_result
                            st.rerun()
                        except Exception as e:
                            st.error(f"번역 중 오류가 발생했습니다: {str(e)}")
                else:
                    st.warning("번역할 텍스트를 입력해주세요.")

    elif choice == "문서 번역":
        st.title("📄 PDF 논문 한국어 번역")
        
        # 1. Define Layout Areas (Top to Bottom)
        header_container = st.container()
        st.markdown("---") # Separator
        settings_container = st.container(border=True)
        result_container = st.container()

        # --- Configuration (Hidden) ---
        # Default paths based on main.py
        layout_ckpt = "/workspace/paper_translation/doclayout_yolo_weight/doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt"
        font_path = "./font/NanumGothicBold.ttf"
        
        # Load Language Config
        lang_config = LangConfig()
        lang_options = list(lang_config.lang_ko_en.keys())

        # --- Settings Area (Bottom) ---
        with settings_container:
            st.markdown("### 📤 PDF 업로드 및 설정")
            uploaded_file = st.file_uploader("번역할 PDF 파일을 선택해주세요", type="pdf")
            
            st.markdown("---")
            l_col1, l_col2 = st.columns(2)
            with l_col1:
                source_lang = st.selectbox("원문 언어 (Source)", options=lang_options, index=lang_options.index("영어"))
            with l_col2:
                target_lang = st.selectbox("번역 언어 (Target)", options=lang_options, index=lang_options.index("한국어"))
            
            st.markdown("---")
            preserve_layout = st.checkbox("문서 구조 유지 (Preserve Layout)", value=False, help="원본 PDF의 레이아웃을 최대한 유지합니다. (체크 해제 시 가독성을 위해 재구성됨)")

        # --- Header Area (Top) ---
        with header_container:
            h_col1, h_col2 = st.columns([3, 1])
            with h_col1:
                st.markdown("""
                <div style='margin-bottom: 0rem;'>
                    PDF 논문을 업로드하면 원본 레이아웃을 최대한 유지하면서 한국어로 번역합니다.<br>
                    이 도구는 <b>Layout Analysis Model</b>과 <b>LLM</b>을 사용하여 정교한 번역을 수행합니다.
                </div>
                """, unsafe_allow_html=True)
            with h_col2:
                # Add some vertical spacing to align button better with text
                st.markdown("<br>", unsafe_allow_html=True)
                # Disable button if no file uploaded
                start_btn = st.button("🚀 번역 시작", disabled=(uploaded_file is None), use_container_width=True)

        # --- Processing Logic ---
        if start_btn:
            if uploaded_file is None:
                st.warning("먼저 PDF 파일을 업로드해주세요.")
            else:
                with result_container:
                    with st.spinner('번역 중입니다... 문서를 분석하고 번역하는 동안 잠시만 기다려주세요.'):
                        # Create a temporary directory for processing
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Save uploaded file to temp path
                            input_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(input_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # Define output paths
                            output_html_name = os.path.splitext(uploaded_file.name)[0] + "_translated.html"
                            output_pdf_name = os.path.splitext(uploaded_file.name)[0] + "_translated.pdf"
                            
                            output_html_path = os.path.join(temp_dir, output_html_name)
                            output_pdf_path = os.path.join(temp_dir, output_pdf_name)

                            try:
                                if preserve_layout:
                                    # Use text_translation.py (Structure Preserved)
                                    paper_translation(
                                        layout_yolo_ckpt=layout_ckpt,
                                        pdf_path=input_path,
                                        source_lang=source_lang,
                                        target_lang=target_lang,
                                        output_pdf_path=output_pdf_path,
                                        font_path=font_path
                                    )
                                else:
                                    # Use html_translation.py (Structure NOT Preserved)
                                    run_html_translation(
                                        layout_ckpt=layout_ckpt,
                                        pdf_path=input_path,
                                        source_lang=source_lang,
                                        target_lang=target_lang,
                                        output_html_path=output_html_path,
                                        output_pdf_path=output_pdf_path,
                                        font_path=font_path
                                    )

                                # Result Section in a Card
                                with st.container(border=True):
                                    st.markdown("### 🎉 번역 완료!")
                                    st.success("번역이 성공적으로 완료되었습니다. 아래 버튼을 눌러 결과를 다운로드하세요.")
                                    st.markdown("---")

                                    # Check if PDF file was created
                                    if os.path.exists(output_pdf_path):
                                        with open(output_pdf_path, "rb") as f:
                                            pdf_data = f.read()
                                        
                                        st.download_button(
                                            label="📥 PDF 다운로드",
                                            data=pdf_data,
                                            file_name=output_pdf_name,
                                            mime="application/pdf",
                                            use_container_width=True
                                        )
                                    else:
                                        st.error("PDF 파일이 생성되지 않았습니다.")

                            except Exception as e:
                                st.error(f"번역 중 오류가 발생했습니다: {str(e)}")
                                st.exception(e)

if __name__ == "__main__":
    main()
