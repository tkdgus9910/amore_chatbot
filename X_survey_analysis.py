# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 11:47:02 2025

@author: tmlab
"""

import pandas as pd 

directory = r"D:\OneDrive\기술인텔리전스\프로젝트\수행중\아모레자문_박상현\설문결과\아모레_멀티모달_설문_v2_인팩_merge.xlsx"

data = pd.read_excel(directory)

data = data.iloc[1:,:(data.shape[1]-1)]

columns = ['내용 정확성 (Content Accuracy)', '완성도 (Completeness)', '간결성(Conciseness)']

for col in columns[:] : 
    data[col] = data[col].apply(lambda x : x.lower())
    data[col] = data[col].apply(lambda x : x.replace(" ", ""))

    data[col] = data[col].str.split('>')
    
    for item in ['a', 'b', 'c']:
        data[item] = data[col].apply(lambda x: 2 - x.index(item))
    

# data['내용 정확성 (Content Accuracy)'] = data['내용 정확성 (Content Accuracy)'].str.split('>')

# data['a'].mean()
# data['b'].mean()
# data['c'].mean()


import seaborn as sns
import matplotlib.pyplot as plt

# 2. ✨ 모델 이름 매핑 및 컬럼명 변경
model_names = {'a': 'Mistral Small 3.2\n24B Instruct',
               'b': 'Gemma 3 27B',
               'c': 'Qwen2.5 VL 32B\nInstruct'}

data = data.rename(columns=model_names)

# Melt
data_melted = data.reset_index().melt(
    id_vars='index', 
    value_vars=list(model_names.values()),
    var_name='item', 
    value_name='score'
)

#%% # 포인트플롯
plt.figure(figsize=(10, 7))
ax = sns.pointplot(
    x='item', y='score', data=data_melted, hue='item',
    palette='viridis', capsize=0.1, legend=False
)

# ✨ --- 평균 점수 어노테이션 추가 시작 --- ✨
# 모델 순서 정의 (x축 순서와 일치하도록)
model_order = ['Mistral Small 3.2\n24B Instruct', 'Gemma 3 27B', 'Qwen2.5 VL 32B\nInstruct']
# 각 모델의 평균 점수 계산
avg_scores = data_melted.groupby('item')['score'].mean().reindex(model_order)

# 각 포인트 위에 텍스트 추가
for i, score in enumerate(avg_scores):
    ax.text(
        i,                # x 좌표 (0, 1, 2)
        score + 0.05,     # y 좌표 (점보다 살짝 위에 표시)
        f'{score:.2f}',   # 표시할 텍스트 (소수점 2자리)
        ha='center',      # 수평 정렬
        fontsize=12,
        fontweight='bold'
    )
# ✨ --- 평균 점수 어노테이션 추가 종료 --- ✨

# 레이블/타이틀
plt.xlabel('Model')
plt.ylabel('Average Score')
plt.title('Average Score Comparison of Models')
plt.ylim(0, 2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

#%%  ✨ --- 막대그래프 생성 --- ✨
plt.figure(figsize=(10, 7))
ax = sns.barplot(
    x='item', y='score', data=data_melted, hue='item',
    palette='viridis', legend=False
)

# ✨ --- 막대 상단에 평균 점수 어노테이션 추가 --- ✨
# 각 막대의 x좌표와 높이(평균 점수)를 가져와서 텍스트를 추가합니다.
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}',      # 표시할 텍스트 (소수점 2자리)
                   (p.get_x() + p.get_width() / 2., p.get_height()), # 텍스트 좌표 (x, y)
                   ha='center', va='center',  # 정렬
                   xytext=(0, 9),             # 텍스트 오프셋
                   textcoords='offset points',
                   fontsize=12,
                   fontweight='bold')


# 레이블/타이틀
plt.xlabel('Model')
plt.ylabel('Average Score')
plt.title('Average Score Comparison of Models')
plt.ylim(0, 2)
plt.grid(True, axis='y', linestyle='--', alpha=0.6) # y축 그리드만 표시
plt.tight_layout()
plt.show()


#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False


# 1. 데이터 불러오기 및 전처리
directory = r"D:\OneDrive\기술인텔리전스\프로젝트\수행중\아모레자문_박상현\설문결과\아모레_멀티모달_설문_v2_인팩_merge.xlsx"
data = pd.read_excel(directory)
data = data.iloc[1:, :(data.shape[1] - 1)]

columns = ['내용 정확성 (Content Accuracy)', '완성도 (Completeness)', '간결성(Conciseness)']
model_keys = ['a', 'b', 'c']

# 데이터 구조 재구성
all_scores_data = []
for col in columns:
    ranking_list = data[col].str.lower().str.replace(" ", "").str.split('>')
    for model in model_keys:
        scores = ranking_list.apply(lambda x: 2 - x.index(model) if model in x else None)
        for score in scores:
            if score is not None:
                all_scores_data.append({
                    'Criterion': col.split(' ')[0],
                    'Model': model,
                    'Score': score
                })
plot_df = pd.DataFrame(all_scores_data)

# 2. 모델 이름 매핑
model_names = {'a': 'Mistral Small 3.2\n24B Instruct',
               'b': 'Gemma 3 27B',
               'c': 'Qwen2.5 VL 32B\nInstruct'}
plot_df['Model'] = plot_df['Model'].map(model_names)

# 3. 그룹 막대 그래프 시각화
plt.figure(figsize=(12, 8))
ax = sns.barplot(
    data=plot_df,
    x='Criterion',
    y='Score',
    hue='Model',
    palette='viridis'
)

# --- ✨ 이 부분이 수정되었습니다 ✨ ---
# 4. 막대 상단에 평균 점수 어노테이션 추가 (0점은 제외)
for p in ax.patches:
    # 막대의 높이가 0보다 클 때만 어노테이션을 추가
    if p.get_height() > 0:
        ax.annotate(f'{p.get_height():.2f}',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center',
                       xytext=(0, 9),
                       textcoords='offset points',
                       fontsize=10,
                       fontweight='bold')
# --- ✨ 여기까지 ✨ ---

# 5. 레이블 및 타이틀 설정
plt.xlabel('Evaluation Criteria (평가 기준)', fontsize=12)
plt.ylabel('Average Score (평균 점수)', fontsize=12)
plt.title('Model Performance Comparison by Criteria (기준별 모델 성능 비교)', fontsize=15, pad=20)
plt.ylim(0, plot_df['Score'].max() * 1.2)
plt.xticks(fontsize=11)
plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()