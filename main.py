import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 페이지 설정 (제목 및 브라우저 탭 아이콘)
st.set_page_config(
    page_title="영화 유형 분석",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기 (K-Means 클러스터링)")
st.markdown("KOBIS 영화 데이터를 바탕으로 주요 속성을 활용해 영화를 지정한 개수의 유형으로 분류하고 시각화하는 앱입니다.")

# 데이터 로드 함수
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 파생 변수 생성
work_df = df.copy()
work_df['log_scrn'] = np.log10(work_df['first_scrn'].replace(0, np.nan))
work_df['log_audi'] = np.log10(work_df['total_audi'].replace(0, np.nan))
work_df['long_run'] = work_df['total_audi'] / work_df['first_week_audi'].replace(0, np.nan)
work_df['long_run'] = work_df['long_run'].clip(upper=20)

feature_map = {
    'log_scrn': '로그 스크린수 (first_scrn)',
    'log_audi': '로그 누적관객 (total_audi)',
    'days_in_top10': '10위권 일수 (days_in_top10)',
    'long_run': '롱런 지수 (long_run)'
}

# 결측치 및 첫 주 관객 0인 영화 제외
required_cols = ['movieNm', 'first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi', 'log_scrn', 'log_audi', 'long_run']
filtered_df = work_df.dropna(subset=required_cols).copy()
filtered_df = filtered_df[filtered_df['first_week_audi'] > 0]

total_count = len(df)
filtered_count = len(filtered_df)

# 전체 편수와 묶은 편수 한 줄 표시
st.info(f"📌 전체 영화 편수: **{total_count:,}편** 중, 유효 데이터로 분석에 사용된 영화 편수: **{filtered_count:,}편**")

# 사이드바: 분석 설정
st.sidebar.header("⚙️ 분석 설정")
default_features = ['log_scrn', 'log_audi', 'days_in_top10', 'long_run']
selected_features = st.sidebar.multiselect(
    "묶는 데 사용할 속성 선택 (2개 이상)",
    options=default_features,
    default=default_features,
    format_func=lambda x: feature_map[x]
)

n_clusters = st.sidebar.slider(
    "묶음 수 선택 (k)",
    min_value=2,
    max_value=7,
    value=3,
    step=1
)

if len(selected_features) < 2:
    st.warning("⚠️ 속성을 2개 이상 선택해 주세요.")
    st.stop()

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(filtered_df[selected_features])

# K-Means 클러스터링
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
filtered_df['cluster_raw'] = kmeans.fit_predict(X_scaled)

# 실루엣 점수 계산
sil_score = silhouette_score(X_scaled, filtered_df['cluster_raw'])

# 묶음 기호 리스트 (최대 7개)
cluster_symbols = ['㉮', '㉯', '㉰', '㉱', '㉲', '㉳', '㉴']

# 누적 관객 평균순으로 기호 부여
cluster_means = filtered_df.groupby('cluster_raw')['total_audi'].mean().reset_index()
cluster_means = cluster_means.sort_values(by='total_audi', ascending=False).reset_index(drop=True)

cluster_label_map = {
    cluster_means.loc[i, 'cluster_raw']: cluster_symbols[i] for i in range(n_clusters)
}

filtered_df['cluster'] = filtered_df['cluster_raw'].map(cluster_label_map)
current_symbols = cluster_symbols[:n_clusters]

# 고른 묶음 수의 실루엣 점수 표시
st.success(f"✨ 현재 선택한 묶음 수(**{n_clusters}개**)의 실루엣 점수(Silhouette Score): **{sil_score:.4f}** (점수는 -1에서 1 사이이며, 1에 가까울수록 묶음이 뚜렷함을 의미합니다.)")

# 2차원 산점도
st.subheader("📊 2차원 영화 유형 산점도")
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("가로축 속성 선택", options=selected_features, format_func=lambda x: feature_map[x], index=0)
with col2:
    default_y_idx = 1 if len(selected_features) > 1 else 0
    y_axis = st.selectbox("세로축 속성 선택", options=selected_features, format_func=lambda x: feature_map[x], index=default_y_idx)

fig_2d = px.scatter(
    filtered_df,
    x=x_axis,
    y=y_axis,
    color='cluster',
    hover_name='movieNm',
    hover_data={'openDt': True, 'genre': True, 'total_audi': ':,', 'first_scrn': ':,', 'days_in_top10': True, 'long_run': ':.2f'},
    labels={**feature_map, 'cluster': '유형'},
    title=f"2차원 영화 유형 분포 (k={n_clusters})",
    category_orders={'cluster': current_symbols}
)
fig_2d.update_layout(height=500)
st.plotly_chart(fig_2d, use_container_width=True)

# 3차원 산점도
st.subheader("🌐 3차원 영화 유형 산점도")
if len(selected_features) >= 3:
    c1, c2, c3 = st.columns(3)
    with c1:
        x_3d = st.selectbox("X축 속성", options=selected_features, format_func=lambda x: feature_map[x], index=0, key='x_3d')
    with c2:
        y_3d = st.selectbox("Y축 속
