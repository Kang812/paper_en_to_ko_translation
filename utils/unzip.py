import zipfile
from glob import glob
from tqdm import tqdm

def extract_zip(zip_path, extract_to):
    # zip 파일 열기
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 모든 파일을 지정한 폴더로 압축 해제
        zip_ref.extractall(extract_to)
        print(f"{zip_path}의 압축을 {extract_to}에 해제했습니다.")

# 사용 예시
if __name__ == '__main__':
    zip_paths = glob('/workspace/paper_translation/paper_translation_data/Broadcast_content_Korean_English_translation_corpus/Official_open_data/Training/labeling/*/*.zip')
    
    #zip_paths = [
    #    '/workspace/paper_translation/paper_translation_data/Broadcast_content_Korean_English_translation_corpus/Official_open_data/Training/labeling/Arts/TL_Korean-Multilingual_koen_Liberal Arts.zip',
    #    '/workspace/paper_translation/paper_translation_data/Broadcast_content_Korean_English_translation_corpus/Official_open_data/Training/labeling/movie_and_drama/TL_Korean-Multilingual_koen_Movie Drama.zip'
    #            ]
    
    for i in tqdm(range(len(zip_paths))):
        zip_path = zip_paths[i]
        filename = zip_path.split("/")[-1]
        output_directory = zip_path.replace(filename, "")
        extract_zip(zip_path, output_directory)
