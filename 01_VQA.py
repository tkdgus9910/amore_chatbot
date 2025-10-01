# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 15:58:48 2025

@author: tmlab
"""
#%% 01. data load 

import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from pathlib import Path
import json
import pandas as pd 

ROOT_DIR = Path("D:/OneDrive/프로젝트/250801_아모레/data")

def process_pdfs_in_directory(root_dir_path: str) -> list:
    """
    지정된 디렉토리와 그 하위의 모든 PDF 파일을 처리하여 페이지별 데이터를 추출합니다.

    Args:
        root_dir_path (str): 검색을 시작할 최상위 디렉토리 경로입니다.

    Returns:
        list: 각 PDF의 모든 페이지에 대한 데이터가 담긴 딕셔너리의 리스트입니다.
              유효한 디렉토리가 아닐 경우 빈 리스트를 반환합니다.
    """
    processed_pages = []
    ROOT_DIR = Path(root_dir_path)

    if not ROOT_DIR.is_dir():
        print(f"오류: '{root_dir_path}'는 유효한 디렉토리가 아닙니다.")
        return []

    # 지정된 경로와 모든 하위 경로에서 .pdf 파일을 찾습니다.
    for pdf_path in ROOT_DIR.rglob("*.pdf"):
        print(f"\n--- '{pdf_path.name}' 파일 처리 시작 ---")
        
        # 파일 경로의 상위 두 폴더 이름을 level1, level2로 사용합니다.
        try:
            level1, level2 = pdf_path.parts[-3:-1]
        except IndexError:
            print(f"  경고: '{pdf_path}'에서 level1, level2를 추출할 수 없습니다. 경로 구조를 확인하세요.")
            level1, level2 = "N/A", "N/A"

        # PyPDF2를 사용하여 메타데이터 추출
        try:
            reader = PdfReader(pdf_path)
            metadata = reader.metadata
        except Exception as e:
            print(f"  경고: '{pdf_path.name}'에서 메타데이터를 읽는 중 오류 발생: {e}")
            metadata = {} # 오류 발생 시 빈 딕셔너리로 초기화

        # PyMuPDF(fitz)를 사용하여 페이지별 텍스트 추출
        doc = None
        try:
            doc = fitz.open(pdf_path)
            
            # 각 페이지를 순회하며 처리
            for i, page in enumerate(doc):
                page_num = i + 1
                
                # 페이지에서 직접 텍스트 추출
                direct_text = page.get_text("text").strip()
                text_size = len(direct_text)
                
                print(f"  - {page_num}/{len(doc)} 페이지 처리 중... (글자 수: {text_size})")

                # 텍스트 길이에 따라 multimodal_applied 플래그 설정
                multimodal_applied = text_size < 1000
                
                # 추출된 데이터와 메타데이터를 딕셔너리 형태로 저장
                page_data = {
                    "level1": level1,
                    "level2": level2,
                    "source": pdf_path.name,
                    "page_number": page_num,
                    "page_count": doc.page_count,
                    "text_recognized": text_size,
                    "multimodal_applied": multimodal_applied,
                    "file_path": str(pdf_path),
                    "direct_text": direct_text,
                    'author': metadata.get('/Author'),
                    'creation_date': metadata.get('/CreationDate')
                }
                processed_pages.append(page_data)
        
        except Exception as e:
            print(f"  오류: '{pdf_path.name}' 파일을 처리하는 중 심각한 오류 발생: {e}")
        finally:
            if doc:
                doc.close()
            print(f"--- '{pdf_path.name}' 파일 처리 완료 ---")

    return processed_pages

# --- 사용 예시 ---
if __name__ == "__main__":
    # 아래 경로를 실제 PDF 파일들이 있는 디렉토리 경로로 수정하여 사용하세요.
    # 예: "C:/Users/YourUser/Documents/Reports"
    target_directory = ROOT_DIR
    
    # # 함수를 호출하여 디렉토리 내 모든 PDF 처리
    all_data = process_pdfs_in_directory(target_directory)

    if all_data:
        print(f"\n\n--- 최종 처리 요약 ---")
        print(f"총 {len(all_data)} 페이지의 데이터를 성공적으로 처리했습니다.")
        
        # 처리된 데이터 중 첫 번째 페이지의 내용을 샘플로 출력
        print("\n첫 번째 페이지 데이터 샘플 (긴 텍스트 내용은 생략):")
        sample_data = all_data[0].copy()
        sample_data['direct_text'] = sample_data['direct_text'][:100] + "..." # 텍스트가 너무 길면 잘라서 표시
        print(json.dumps(sample_data, indent=4, ensure_ascii=False))
        
df = pd.DataFrame(all_data)

#%% 02. EDA - 문서 별 텍스트 길이 분포 확인 

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 이제 라이브러리를 불러옵니다.
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

temp = df.groupby(['level2', 'source'])['text_recognized'].mean()

# 이후 시각화 코드를 실행합니다.
sns.set_theme()
sns.histplot(temp, kde=True, color="red")
plt.show()

temp = df.groupby(['level2', 'source'])['text_recognized'].mean()

# 이후 시각화 코드를 실행합니다.
sns.set_theme()

sns.histplot(temp, bins= 15, kde=True, color="blue")

plt.show()

df['multimodal_applied'] = False  
source_dict = dict(zip(df['source'], df['text_recognized']))
source_list = [k for k,v in source_dict.items() if v < 2000] # 2000자 미만이면 멀티모달 적용

df['multimodal_applied'] = df['source'].apply(lambda x : True if x in source_list else False) 

#%% 03. API+VQA 기반 이미지 텍스트 처리

import os
import io
import base64
import requests
from pdf2image import convert_from_path
from PIL import Image

# =========================
# 0) 환경설정
# =========================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # 환경변수로 보관 권장
if not OPENROUTER_API_KEY:
    raise RuntimeError("환경변수 OPENROUTER_API_KEY 를 설정하세요.")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL = "mistralai/mistral-small-3.2-24b-instruct" # ok
# MODEL = "qwen/qwen2.5-vl-32b-instruct" # ok

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# =========================
# 1) 입력 경로
# =========================
pdf_path = r"D:\OneDrive\프로젝트\250801_아모레\data\1. 화장품 산업 동향\1-1. 소비자 트렌드\(1)[PlayD] VOICE Trend_12월_뷰티_fin.pdf"
poppler_bin_path = r"C:\poppler-25.07.0\Library\bin"

# =========================
# 2) 유틸: PIL 이미지 → data URL (base64)
# =========================
def pil_to_data_url(img: Image.Image, fmt="PNG", max_side=1800) -> str:
    """
    모델 입력에 과도하게 큰 이미지는 성능/속도 저하를 유발할 수 있어
    한쪽 변을 max_side로 리사이즈(축소)한 후 base64 data URL로 변환합니다.
    """
    img = img.convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

# =========================
# 3) OpenRouter 호출 함수
# =========================
def ask_openrouter_vision(model: str, question: str, image_data_url: str, temperature: float = 0.0) -> str:
    """
    OpenAI 호환 chat/completions 형식으로 비전-언어 모델에 질의.
    image_url에는 data URL(base64)을 전달.
    """
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant for document visual question answering. "
                           "Read the document image carefully and answer concisely in Korean."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]
            }
        ]
    }
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

# ===============================================
# 4) 파이프라인 함수: 특정 페이지 리스트 처리
# ===============================================
def run_doc_vqa_on_pages(
    pdf_path: str,
    poppler_path: str,
    page_numbers: list[int],
    model: str = MODEL,
    question: str = "이 페이지의 주요 내용은 무엇인가요?",
) -> dict[int, str]:
    """
    지정된 페이지 번호 리스트에 대해서만 VQA를 수행합니다.

    Args:
        pdf_path (str): 분석할 PDF 파일 경로.
        poppler_path (str): Poppler 바이너리 경로.
        page_numbers (list[int]): 분석할 페이지 번호 리스트 (예: [1, 3, 5]).
        model (str): 사용할 비전-언어 모델 이름.
        question (str): 각 페이지에 던질 질문.

    Returns:
        dict[int, str]: 페이지 번호를 key로, 모델의 답변을 value로 갖는 딕셔너리.
                        오류 발생 시 값은 "error"가 됩니다.
    """
    if not os.path.exists(pdf_path):
        print(f"오류: PDF 파일을 찾을 수 없습니다. 경로를 확인하세요: {pdf_path}")
        return {}

    try:
        # 전체 페이지를 한번에 이미지로 변환
        images = convert_from_path(pdf_path, poppler_path=poppler_path)
    except Exception as e:
        print(f"PDF→이미지 변환 중 오류: {e}")
        print("Poppler 경로(poppler_path)가 올바른지 확인하세요.")
        return {}

    total_pages = len(images)
    print(f"총 {total_pages} 페이지의 PDF 파일을 이미지로 변환했습니다.")
    print(f"요청된 페이지: {page_numbers}")
    print("-" * 40)

    answer_dict = {}

    # 지정된 페이지 번호 리스트를 순회
    for page_num in page_numbers:
        # 페이지 번호는 1부터, 리스트 인덱스는 0부터 시작하므로 변환
        idx = page_num - 1

        # 유효한 페이지 번호인지 확인
        if not (0 <= idx < total_pages):
            print(f"⚠️ 경고: {page_num}번 페이지는 유효한 범위(1-{total_pages})를 벗어나므로 건너뜁니다.")
            print("-" * 40)
            continue

        print(f"📄 {page_num} 페이지 분석 중...")
        try:
            data_url = pil_to_data_url(images[idx], fmt="PNG", max_side=1800)
            answer = ask_openrouter_vision(model=model, question=question, image_data_url=data_url)
            print(f"질문: {question}")
            print(f"답변: {answer}")
            answer_dict[page_num] = answer

        except requests.HTTPError as http_err:
            print(f"[HTTP 오류] 페이지 {page_num}: {http_err} | 응답: {getattr(http_err, 'response', None)}")
            answer_dict[page_num] = "error"
        except Exception as e:
            print(f"[예외] 페이지 {page_num}: {e}")
            answer_dict[page_num] = "error"
        finally:
            print("-" * 40)

    return answer_dict



#%% 04. 샘플 데이터 준비

sample_list = ["(2024-08-26) EUROPE 5개 국가에 대한 사업 영역 분석",
 "[24.09] 경쟁사 신제품 조사 (선케어)", 
 "[기술 동향 보고서]_자외선 차단 연구 기술 동향 보고서_vf", 
 "미래기술융합화장품_20220121_이성원",
 "1. 맞춤형화장품 관련 법령_배포", 
 "5.+광고+사전자문+및+모니터링+업무+안내+", 
 "9.+2025년+화장품+정책설명회+발표자료_표시광고+위반사례", 
 "20250124_2024년+하반기+기능성화장품+심사+및+보고+현황",
 "기능성 화장품",
 "pwc K-뷰티 산업의 변화",]

sample_list = [i+".pdf" for i in sample_list]

df_sample = df.loc[df['source'].apply(lambda x : True if x in sample_list else False),:]

# 핵심 페이지 추출

rep_pages = {}
rep_pages[sample_list[0]] = [2,4,5,8,10]
rep_pages[sample_list[1]] = [2,4,8,15,20]
rep_pages[sample_list[2]] = [1,4,6,12,32]
rep_pages[sample_list[3]] = [5,12,31,46,64]
rep_pages[sample_list[4]] = [8,10,12,16,27]
rep_pages[sample_list[5]] = [3,4,6,10,11] #광고 사전자문 
rep_pages[sample_list[6]] = [2,5,7,12,16] #20250124 
rep_pages[sample_list[7]] = [1,2,3] #20250124
rep_pages[sample_list[8]] = [5,15,22,37,39] #기능성
rep_pages[sample_list[9]] = [3,5,13,21,31] #기능성

df_sample_pages = pd.DataFrame()

for file, pages in rep_pages.items() :
    print(file)
    temp_df = df_sample.loc[df_sample['source'] == file, : ]
    temp_df = temp_df.loc[temp_df['page_number'].isin(pages), : ]
    
    df_sample_pages= pd.concat([df_sample_pages, temp_df], axis = 0)

df_sample_ = df_sample.drop_duplicates('source')
file_directory_dict = dict(zip(df_sample_['source'], df_sample_['file_path']))

df_sample_pages = df

#%% 05. iteration

# MODEL = "mistralai/mistral-small-3.2-24b-instruct" # ok
MODEL = "google/gemma-3-27b-it" # ok
# MODEL = "qwen/qwen2.5-vl-32b-instruct" # ok
# MODEL = "meta-llama/llama-4-scout" # ok
# MODEL = "meta-llama/llama-4-maverick" # ok

df_sample_pages['VQA_result_{}'.format(MODEL)] = np.nan

prompt = """당신은 문서·슬라이드·웹페이지 스크린샷을 읽는 VQA 분석가입니다.
이 한 페이지에서 (1) 핵심 메시지 3줄과 (2) 핵심 수치 Top-N 을 뽑아주세요.

규칙:
- 보이는 정보만 사용, 추정/상상 금지. 없으면 "미확인".
- 수치는 {라벨, 값, 단위, 기간/기준, 근거영역}으로 정리.
- 값은 원문 단위 유지(환산 금지).
- 충돌 수치가 있으면 더 두드러진 위치(헤드라인/굵은글/요약/타일/표합계) 우선, 나머지는 "메모"에 기록.

출력(한국어):
핵심요약:
- (요약1)
- (요약2)
- (요약3)

핵심수치(상위 N=5):
1) 라벨: … | 값: … | 단위: … | 기간/기준: … | 근거영역: …
2) …
메모: (필요 시만)

"""

df_sample_pages = df_sample_pages.reset_index(drop = 1)

# =========================
# 5) 실행 예시
# =========================
if __name__ == "__main__":
    
    pdf_path_list = set(df_sample_pages['file_path'])
    
    for pdf_path in pdf_path_list : 

        # pdf_path = r"D:\OneDrive\프로젝트\250801_아모레\data\2. 생산기술 트렌드\2-5. 융합기술\미래기술융합화장품_20220121_이성원.pdf"
    
        df_temp = df_sample_pages.loc[df_sample_pages['file_path'] == pdf_path , : ]
        
        target_pages = list(df_temp['page_number'])
        
        if str(list(df_temp['VQA_result_{}'.format(MODEL)])[0]) != "nan" : 
            print(pdf_path.split('\\')[-1])
            continue
        
        else :
                
            result = run_doc_vqa_on_pages(
                pdf_path=pdf_path,
                poppler_path=poppler_bin_path,
            page_numbers=target_pages,
                model = MODEL,
                question= prompt,)
            
            mask = df_sample_pages['file_path'].eq(pdf_path)
            df_sample_pages.loc[mask, 'VQA_result_{}'.format(MODEL)] = list(result.values())  # result가 스칼라면 전 행에 브로드캐스트


#%% 06. 저장

import pickle

with open("df_sample_pages.pkl", "wb") as f :
    pickle.dump(df_sample_pages, f)
    

output_directory = "D:/OneDrive/프로젝트/250801_아모레/설문용샘플/1_VQA/"

# 문자열 열에 한해서 \x01만 우선 제거/치환
for col in df_sample_pages.columns:
    if df_sample_pages[col].dtype == "object":
        df_sample_pages[col] = df_sample_pages[col].str.replace("\x01", " ", regex=False)

df_sample_pages.to_excel(output_directory + "result_test.xlsx")
