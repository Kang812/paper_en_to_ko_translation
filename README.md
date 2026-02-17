# 📄 PDF Paper Translation to Korean (논문 번역 앱) - Extended

> **원본 레이아웃을 그대로 유지하며, PDF 논문을 한국어로 정교하게 번역해주는 AI 도구입니다.**  
> Layout Analysis Model과 LLM(Large Language Model)을 결합하여 가독성 높은 번역 결과를 제공합니다.

---

## ✨ Key Features (주요 기능)

### 1. 🖼️ Layout Preservation Options (레이아웃 유지 옵션)
-   **Preserve Layout (구조 유지)**: 원본 PDF의 단락, 표, 그림 위치를 그대로 유지하며 텍스트만 번역합니다. (이미지 위 덮어쓰기 방식)
-   **Reformat (재구성)**: 가독성을 최우선으로 하여, 줄글 형태의 깔끔한 HTML 기반 PDF로 재구성합니다.

### 2. 🤖 AI Translator (AI 번역기)
-   논문 전체 번역 외에도, **텍스트를 직접 입력하여 빠르게 번역**할 수 있는 독립형 번역기 기능을 제공합니다.
-   **Language Selection**: 영어 -> 한국어 뿐만 아니라 다양한 언어 쌍을 지원할 수 있는 확장성을 갖췄습니다 (현재 기본: 영/한).

### 3. 🧠 Enhanced Processing (향상된 처리)
-   **LaTeX Math Support**: 논문 내의 LaTeX 수식을 감지하여 깨지지 않고 유니코드로 변환하여 번역 품질을 높였습니다.
-   **Overlap Prevention**: 기존의 텍스트 겹침 문제를 해결하기 위해 정교한 여백(Margin) 계산 알고리즘을 적용했습니다.

### 4. 🎨 Modern & User-Friendly UI
-   **Streamlit** 기반의 직관적인 웹 인터페이스.
-   **Language Selector**: 원문 언어와 번역 언어를 자유롭게 선택 가능.
-   **PDF Only Download**: 가장 호환성이 좋은 PDF 포맷으로 결과를 제공합니다.

---

## 🛠️ Menu Structure (메뉴 구성)

| 메뉴 (Menu) | 아이콘 | 설명 |
| :--- | :---: | :--- |
| **번역기** | `translate` | 텍스트를 직접 입력하여 즉시 번역하는 AI 번역 도구 |
| **문서 번역** | `file-text` | **[메인 기능]** PDF 파일을 업로드하고, 레이아웃 유지 여부를 선택하여 번역 |

---

## 🚀 Installation & Usage (설치 및 실행)

### 1. Prerequisites (필수 요구사항)

이 프로젝트는 Python 3.8+ 환경에서 실행됩니다. 필요한 라이브러리를 설치해주세요.

```bash
# 필수 의존성 설치
pip install -r requirements.txt
```

### 2. Run the Application (앱 실행)

```bash
streamlit run app.py
```

### 3. How to Use (사용 방법)

#### 📄 문서 번역 (Document Translation)
1.  앱 실행 후 사이드바에서 **`문서 번역`** 메뉴 선택.
2.  **`PDF 파일 선택`** 영역에 논문 업로드.
3.  **원문/번역 언어** 선택.
4.  **문서 구조 유지** 체크박스:
    -   ✅ 체크 시: 원본 모양 그대로 유지 (이미지 처리).
    -   ⬜ 체크 해제 시: 깔끔한 줄글 형태로 변환.
5.  **`🚀 번역 시작`** 클릭 후 완료되면 PDF 다운로드.

#### 🤖 번역기 (Translator)
1.  사이드바에서 **`번역기`** 메뉴 선택.
2.  왼쪽 입력창에 번역하고 싶은 텍스트 입력.
3.  언어 설정 확인 후 **`🚀 번역하기`** 클릭.
4.  오른쪽 창에서 결과 확인.

---

## ⚙️ Configuration (설정)

-   **Wrapper Model**: `DocLayout-YOLO` (Layout Analysis)
-   **Translation Model**: `Ollama` (gemma:12b etc.)
-   **Processing**: `pdfplumber` (Font extraction), `pylatexenc` (Math conversion)

---

## 📂 Project Structure

```
paper_translation/
├── app.py                # Streamlit 메인 애플리케이션
├── utils/
│   ├── html_translation.py # HTML 기반 재구성 번역 로직
│   ├── text_translation.py # 레이아웃 유지(Drawing) 번역 로직
│   ├── translation.py      # 텍스트 단순 번역 로직
│   └── lang_config.py      # 언어 설정 관리
├── requirements.txt      # 의존성 목록
└── ...
```

---

## 📝 License

This project is for educational and research purposes.

---

<p align="center">
    Created with ❤️ by <b>Paper Translation Team</b>
</p>
