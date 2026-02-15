# 📄 PDF Paper Translation to Korean (논문 번역 앱)

> **원본 레이아웃을 그대로 유지하며, PDF 논문을 한국어로 정교하게 번역해주는 AI 도구입니다.**  
> Layout Analysis Model과 LLM(Large Language Model)을 결합하여 가독성 높은 번역 결과를 제공합니다.

---

## ✨ Key Features (주요 기능)

### 1. 🖼️ Layout Preservation (레이아웃 유지)
-   단순히 텍스트만 추출하여 번역하는 것이 아니라, **원본 PDF의 단락, 표, 그림 위치 등 레이아웃 구조를 분석 및 유지**합니다.
-   번역된 결과물도 논문 형식 그대로의 HTML/PDF로 제공됩니다.

### 2. 🧠 Advanced Translation (정교한 번역)
-   **Layout Analysis**: `DocLayout-YOLO` 모델을 사용하여 문서 구조를 정밀하게 파악합니다.
-   **LLM Integration**: 문맥을 이해하는 LLM을 통해 자연스러운 한국어 번역을 제공합니다.

### 3. 🎨 Modern & User-Friendly UI
-   **Streamlit** 기반의 직관적인 웹 인터페이스.
-   **Custom CSS** & **Pretendard Font**: 깔끔하고 세련된 디자인과 가독성 높은 폰트 적용.
-   **Interactive Menu**: `streamlit-option-menu`를 활용한 직관적인 사이드바 내비게이션.

---

## 🛠️ Menu Structure (메뉴 구성)

| 메뉴 (Menu) | 아이콘 | 설명 |
| :--- | :---: | :--- |
| **번역기** | `translate` | (준비 중) 텍스트 전용 번역 기능 |
| **문서 번역(단순)** | `file-text` | **[메인 기능]** PDF 파일을 업로드하여 레이아웃을 유지하며 번역 |
| **문서 번역(문맥 이해)** | `book` | (준비 중) 전체 논문의 맥락을 더 깊이 이해하는 고급 번역 |

---

## 🚀 Installation & Usage (설치 및 실행)

### 1. Prerequisites (필수 요구사항)

이 프로젝트는 Python 3.8+ 환경에서 실행됩니다. 필요한 라이브러리를 설치해주세요.

```bash
# 기본 의존성 설치 (requirements.txt가 있다면)
pip install -r requirements.txt

# UI 구성을 위한 추가 라이브러리 설치
pip install streamlit streamlit-option-menu
```

### 2. Run the Application (앱 실행)

```bash
streamlit run app.py
```

### 3. How to Use (사용 방법)
1.  앱이 실행되면 왼쪽 사이드바에서 **`문서 번역(단순)`** 메뉴를 선택합니다.
2.  **`PDF 파일 선택`** 영역에 번역할 논문 PDF 파일을 드래그하거나 업로드합니다.
3.  **`🚀 번역 시작`** 버튼을 클릭합니다.
4.  번역이 완료되면 결과 화면에서 **HTML** 또는 **PDF** 파일을 다운로드할 수 있습니다.

---

## ⚙️ Configuration (설정)

현재 버전('문서 번역-단순')에서는 사용자 편의를 위해 복잡한 설정을 숨기고, 최적화된 기본값을 사용합니다.

-   **Wrapper Model**: `DocLayout-YOLO` (Layout Analysis)
-   **Translation Model**: Local LLM Checkpoint or Ollama
-   **Font**: `NanumGothicBold` (for PDF generation)

---

## 📂 Project Structure

```
paper_translation/
├── app.py                # Streamlit 메인 애플리케이션
├── utils/
│   └── html_translation.py # 핵심 번역 로직 (HTML/PDF 생성)
├── doclayout_yolo_weight/ # 레이아웃 분석 모델 가중치
├── save_model/           # LLM 모델 체크포인트
├── font/                 # PDF 생성용 폰트
└── ...
```

---

## 📝 License

This project is for educational and research purposes.

---

<p align="center">
    Created with ❤️ by <b>Paper Translation Team</b>
</p>
