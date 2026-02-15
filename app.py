
import os
import streamlit as st
import tempfile
import argparse
import shutil
from utils.html_translation import run_html_translation
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
        choice = option_menu("메뉴", ["번역기", "문서 번역(단순)", "문서 번역(문맥 이해)"],
                             icons=['translate', 'file-text', 'book'],
                             menu_icon="cast", default_index=1,
                             styles={
                                 "container": {"padding": "5!important", "background-color": "#ffffff"},
                                 "icon": {"color": "#02ab21", "font-size": "20px"}, 
                                 "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px", "--hover-color": "#f0f2f6", "font-family": "Pretendard"},
                                 "nav-link-selected": {"background-color": "#e8f5e9", "color": "#02ab21", "font-weight": "600"},
                             }
                            )

    if choice == "번역기":
        st.title("번역기")
        st.info("준비 중입니다!")

    elif choice == "문서 번역(단순)":
        st.title("📄 PDF 논문 한국어 번역")
        st.markdown("""
        <div style='margin-bottom: 2rem;'>
            PDF 논문을 업로드하면 원본 레이아웃을 최대한 유지하면서 한국어로 번역합니다.<br>
            이 도구는 <b>Layout Analysis Model</b>과 <b>LLM</b>을 사용하여 정교한 번역을 수행합니다.
        </div>
        """, unsafe_allow_html=True)

        # --- Configuration (Hidden) ---
        # Default paths based on main.py
        layout_ckpt = "/workspace/paper_translation/doclayout_yolo_weight/doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt"
        llm_ckpt = "/workspace/paper_translation/save_model/checkpoint-34951"
        font_path = "./font/NanumGothicBold.ttf"
        ollama_mode = False

        # --- Main Area ---
        # Using st.container(border=True) to create card-like sections
        with st.container(border=True):
            st.markdown("### 📤 PDF 업로드")
            uploaded_file = st.file_uploader("번역할 PDF 파일을 선택해주세요", type="pdf")

        if uploaded_file is not None:
            # Add some spacing
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                start_btn = st.button("🚀 번역 시작")

            if start_btn:
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
                            # Run the translation
                            run_html_translation(
                                layout_ckpt=layout_ckpt,
                                llm_ckpt=llm_ckpt,
                                pdf_path=input_path,
                                output_html_path=output_html_path,
                                output_pdf_path=output_pdf_path,
                                font_path=font_path,
                                ollama_mode=ollama_mode
                            )

                            # Result Section in a Card
                            with st.container(border=True):
                                st.markdown("### 🎉 번역 완료!")
                                st.success("번역이 성공적으로 완료되었습니다. 아래 버튼을 눌러 결과를 다운로드하세요.")
                                st.markdown("---")

                                # Create columns for download buttons
                                d_col1, d_col2 = st.columns(2)

                                # Check if files were created
                                if os.path.exists(output_html_path):
                                    with open(output_html_path, "rb") as f:
                                        html_data = f.read()
                                    d_col1.download_button(
                                        label="📥 HTML 다운로드",
                                        data=html_data,
                                        file_name=output_html_name,
                                        mime="text/html"
                                    )
                                else:
                                    d_col1.error("HTML 파일이 생성되지 않았습니다.")

                                if os.path.exists(output_pdf_path):
                                    with open(output_pdf_path, "rb") as f:
                                        pdf_data = f.read()
                                    d_col2.download_button(
                                        label="📥 PDF 다운로드",
                                        data=pdf_data,
                                        file_name=output_pdf_name,
                                        mime="application/pdf"
                                    )
                                else:
                                    d_col2.error("PDF 파일이 생성되지 않았습니다.")

                        except Exception as e:
                            st.error(f"번역 중 오류가 발생했습니다: {str(e)}")
                            st.exception(e)

    elif choice == "문서 번역(문맥 이해)":
        st.title("문서 번역(문맥 이해)")
        st.info("준비 중입니다!")

if __name__ == "__main__":
    main()
