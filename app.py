
import os
import streamlit as st
import tempfile
import argparse
import shutil
from utils.html_translation import run_html_translation

# Set page configuration
st.set_page_config(
    page_title="Paper Translation App",
    page_icon="📄",
    layout="wide"
)

def main():
    st.title("📄 PDF Paper Translation to Korean")
    st.markdown("""
    Upload a PDF paper to translate it into Korean while preserving the layout.
    This tool uses a layout analysis model and an LLM for translation.
    """)

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")
    
    # Default paths based on main.py
    default_layout_ckpt = "/workspace/paper_translation/doclayout_yolo_weight/doclayout_yolo_doclaynet_imgsz1120_docsynth_pretrain.pt"
    default_llm_ckpt = "/workspace/paper_translation/save_model/checkpoint-34951"
    default_font_path = "./font/NanumGothicBold.ttf"

    layout_ckpt = st.sidebar.text_input("Layout YOLO Checkpoint", value=default_layout_ckpt)
    
    ollama_mode = st.sidebar.checkbox("Use Ollama Mode", value=False)
    
    if ollama_mode:
        llm_ckpt = "ollama mode"
        st.sidebar.info("Using Ollama for translation.")
    else:
        llm_ckpt = st.sidebar.text_input("LLM Checkpoint Path", value=default_llm_ckpt)

    font_path = st.sidebar.text_input("Font Path", value=default_font_path)

    # --- Main Area ---
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        st.success("File uploaded successfully!")

        if st.button("Start Translation"):
            with st.spinner('Translating... This may take a while depending on the file size.'):
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
                        # Note: We are capturing stdout/stderr might be needed for real-time logs, 
                        # but for simplicity we just run it.
                        run_html_translation(
                            layout_ckpt=layout_ckpt,
                            llm_ckpt=llm_ckpt,
                            pdf_path=input_path,
                            output_html_path=output_html_path,
                            output_pdf_path=output_pdf_path,
                            font_path=font_path,
                            ollama_mode=ollama_mode
                        )

                        st.success("Translation Complete!")

                        # Create columns for download buttons
                        col1, col2 = st.columns(2)

                        # Check if files were created
                        if os.path.exists(output_html_path):
                            with open(output_html_path, "rb") as f:
                                html_data = f.read()
                            col1.download_button(
                                label="Download HTML",
                                data=html_data,
                                file_name=output_html_name,
                                mime="text/html"
                            )
                        else:
                            col1.error("HTML file was not generated.")

                        if os.path.exists(output_pdf_path):
                            with open(output_pdf_path, "rb") as f:
                                pdf_data = f.read()
                            col2.download_button(
                                label="Download PDF",
                                data=pdf_data,
                                file_name=output_pdf_name,
                                mime="application/pdf"
                            )
                        else:
                            col2.error("PDF file was not generated. (Check logs/font settings)")

                    except Exception as e:
                        st.error(f"An error occurred during translation: {str(e)}")
                        st.exception(e)

if __name__ == "__main__":
    main()
