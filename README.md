
## 환경 setting
```
pip install -r requirements.txt
pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

## 사용 데이터(AI 허브 데이터 사용)
- [전문분야 한영 말뭉치](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=111)
- [한국어-영어 번역 말뭉치(사회과학)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=125)
- [한국어-영어 번역 말뭉치(기술과학)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=124)
- [기술과학 분야 한-영 번역 병렬 말뭉치 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71266)
- [방송콘텐츠 한국어-영어 번역 말뭉치](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71382)
- [한국어-다국어 번역 말뭉치(기초과학)](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71496)

- 한국어 -> 영어 데이터 만 사용해서 모델 학습

## 데이터 셋 전처리
- ./utils/json_to_df.py : json을 사용해서 dataframe으로 변환
- ./utils/convert_dataset.py : huggingface에 맞는 데이터 셋으로 변환

## LLM 학습
```
./train.sh
```

## 영어 PDF를 한국어로 변환
```
./exec.sh
```