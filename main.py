import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 페이지 설정 (제목 및 브라우저 탭 아이콘)
st.set_page_config(
    page_title="영화 유형 분석",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기 (K-Means 클러스터링)")
st.markdown("KOBIS 영화 데이터를 바탕으로 주요 속성을 활용해 영화를 3가지 유형으로 분류하고 시각화하는 앱입니다.")

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

# 사이드바: 클러스터링 속성 선택
st.sidebar.header("⚙️ 분석 설정")
default_features = ['log_scrn', 'log_audi', 'days_in_top10', 'long_run']
selected_features = st.sidebar.multiselect(
    "묶는 데 사용할 속성 선택 (2개 이상)",
    options=default_features,
    default=default_features,
    format_func=lambda x: feature_map[x]
)

if len(selected_features) < 2:
    st.warning("⚠️ 속성을 2개 이상 선택해 주세요.")
    st.stop()

# 표준화 및 K-Means 클러스터링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(filtered_df[selected_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
filtered_df['cluster_raw'] = kmeans.fit_predict(X_scaled)

# 묶음 번호 레이블링: 누적 관객(total_audi) 평균이 큰 순서대로 ㉮, ㉯, ㉰ 부여
cluster_means = filtered_df.groupby('cluster_raw')['total_audi'].mean().reset_index()
cluster_means = cluster_means.sort_values(by='total_audi', ascending=False).reset_index(drop=True)

cluster_label_map = {
    cluster_means.loc[0, 'cluster_raw']: '㉮',
    cluster_means.loc[1, 'cluster_raw']: '㉯',
    cluster_means.loc[2, 'cluster_raw']: '㉰'
}

filtered_df['cluster'] = filtered_df['cluster_raw'].map(cluster_label_map)

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
    title=f"2차원 영화 유형 분포",
    category_orders={'cluster': ['㉮', '㉯', '㉰']}
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
        y_3d = st.selectbox("Y축 속성", options=selected_features, format_func=lambda x: feature_map[x], index=1, key='y_3d')
    with c3:
        z_3d = st.selectbox("Z축 속성", options=selected_features, format_func=lambda x: feature_map[x], index=2, key='z_3d')

    fig_3d = px.scatter_3d(
        filtered_df,
        x=x_3d,
        y=y_3d,
        z=z_3d,
        color='cluster',
        hover_name='movieNm',
        hover_data={'total_audi': ':,', 'first_scrn': ':,', 'days_in_top10': True, 'long_run': ':.2f'},
        labels={**feature_map, 'cluster': '유형'},
        title=f"3차원 영화 유형 분포",
        category_orders={'cluster': ['㉮', '㉯', '㉰']}
    )
    fig_3d.update_traces(marker=dict(size=3, opacity=0.8))
    fig_3d.update_layout(height=650)
    st.plotly_chart(fig_3d, use_container_width=True)
else:
    st.info("ℹ️ 3차원 산점도를 그리려면 클러스터링에 사용된 속성을 **3개 이상** 선택해 주세요.")

# 유형별 통계 요약 표
st.subheader("📋 묶음별 편수 및 속성 평균 (원래 단위)")

summary_data = []
for cluster in ['㉮', '㉯', '㉰']:
    c_df = filtered_df[filtered_df['cluster'] == cluster]
    count = len(c_df)
    mean_scrn = c_df['first_scrn'].mean()
    mean_audi = c_df['total_audi'].mean()
    mean_days = c_df['days_in_top10'].mean()
    mean_long = c_df['long_run'].mean()
    
    summary_data.append({
        '유형': f"유형 {cluster}",
        '영화 편수': f"{count:,}편",
        '평균 스크린 수': f"{mean_scrn:,.1f}개",
        '평균 누적 관객': f"{mean_audi:,.0f}명",
        '평균 10위권 일수': f"{mean_days:,.1f}일",
        '평균 롱런 지수': f"{mean_long:,.2f}"
    })

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True)

# 유형별 대표 영화 TOP 5
st.subheader("🏆 묶음별 누적 관객 TOP 5 영화")

cols = st.columns(3)
clusters = ['㉮', '㉯', '㉰']

for idx, cluster in enumerate(clusters):
    with cols[idx]:
        st.markdown(f"#### 유형 {cluster}")
        top5 = filtered_df[filtered_df['cluster'] == cluster].sort_values(by='total_audi', ascending=False).head(5)
        for i, row in enumerate(top5.iterrows(), 1):
            movie = row[1]
            st.markdown(f"**{i}. {movie['movieNm']}**")
            st.caption(f"누적: {movie['total_audi']:,}명 / 스크린: {movie['first_scrn']:,}개")
