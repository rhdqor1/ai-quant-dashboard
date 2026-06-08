import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

# 1. 페이지 기본 설정
st.set_page_config(layout="wide", page_title="글로벌 퀀트 대시보드")
st.title("🌏 한·미·중 동조화 기반 AI 퀀트 대시보드")
st.markdown("과거 6년간의 미국(QQQ), 한국(KODEX 200), 중국(ASHR) 데이터를 학습하여 각국의 상승 확률과 방향성을 예측합니다.")

# 2. 데이터 로드 (캐싱)
@st.cache_data
def load_data():
    start_date = "2020-01-01"
    end_date = "2026-06-07"
    us_etf = yf.Ticker("QQQ").history(start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'US_QQQ'})
    kr_etf = yf.Ticker("069500.KS").history(start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'KR_KODEX'})
    cn_etf = yf.Ticker("ASHR").history(start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'CN_ASHR'})

    us_etf.index = us_etf.index.tz_localize(None)
    kr_etf.index = kr_etf.index.tz_localize(None)
    cn_etf.index = cn_etf.index.tz_localize(None)

    df = us_etf.join(kr_etf, how='outer').join(cn_etf, how='outer')
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

df = load_data()

# 3. 모델 학습 (3개국 각각의 내일 오를 확률 계산 + 미국 시그널 데이터)
@st.cache_resource
def train_models(data_values):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_values)
    window_size = 10
    
    X = []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size : i])
    X = np.array(X)
    
    results = {}
    names = ['US_QQQ', 'KR_KODEX', 'CN_ASHR']
    
    # 3개국 각각에 대해 빠르게 AI 모델을 학습시켜 '내일 확률'을 뽑아냅니다.
    for col_idx, name in enumerate(names):
        y = [1 if scaled_data[i, col_idx] > scaled_data[i-1, col_idx] else 0 for i in range(window_size, len(scaled_data))]
        y = np.array(y)
        
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # 빠른 학습을 위해 LSTM 층을 1개로 경량화
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='sigmoid'))
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        # 오늘(최근 10일) 데이터를 넣어서 내일 확률 예측
        latest_X = np.array([scaled_data[-window_size:]])
        next_prob = model.predict(latest_X, verbose=0)[0][0]
        results[f"{name}_prob"] = next_prob
        
        # 미국(QQQ)은 하단 시그널 차트를 위해 전체 테스트 예측값 저장
        if name == 'US_QQQ':
            results['us_pred_prob'] = model.predict(X_test, verbose=0)
            results['us_y_test'] = y_test
            
    return results

with st.spinner('🤖 AI가 3개국 각각의 상승 확률을 정밀 계산 중입니다... (약 15초 소요)'):
    ai_results = train_models(df.values)

st.markdown("---")
st.subheader("📊 글로벌 3대 시장 주도주 흐름 & AI 단기 예측")

# 4. 상단: 3국 지수 시각화 (Plotly 인터랙티브 차트 + 확률 뱃지)
col1, col2, col3 = st.columns(3)

# 차트 그리는 함수 (중복 코드 방지)
def draw_plotly_chart(dataframe, column_name, color):
    fig = px.line(dataframe, y=column_name, color_discrete_sequence=[color])
    fig.update_layout(
        xaxis_title="", yaxis_title="", 
        margin=dict(l=0, r=0, t=10, b=0), 
        height=250,
        hovermode="x unified" 
    )
    return fig

with col1:
    st.markdown("### 🇺🇸 미국 (QQQ)")
    prob = ai_results['US_QQQ_prob'] * 100
    st.info(f"🔮 AI 내일 상승 확률: **{prob:.1f}%**")
    # 💡 여기에 config={'scrollZoom': True} 옵션이 추가되었습니다.
    st.plotly_chart(draw_plotly_chart(df, 'US_QQQ', '#1f77b4'), use_container_width=True, config={'scrollZoom': True})

with col2:
    st.markdown("### 🇰🇷 한국 (KODEX 200)")
    prob = ai_results['KR_KODEX_prob'] * 100
    st.warning(f"🔮 AI 내일 상승 확률: **{prob:.1f}%**")
    st.plotly_chart(draw_plotly_chart(df, 'KR_KODEX', '#ff7f0e'), use_container_width=True, config={'scrollZoom': True})

with col3:
    st.markdown("### 🇨🇳 중국 (ASHR)")
    prob = ai_results['CN_ASHR_prob'] * 100
    st.success(f"🔮 AI 내일 상승 확률: **{prob:.1f}%**")
    st.plotly_chart(draw_plotly_chart(df, 'CN_ASHR', '#2ca02c'), use_container_width=True, config={'scrollZoom': True})

st.markdown("---")

# 5. 하단: HTS 스타일 시그널 차트 (미국 QQQ 기준)
st.subheader("🔥 딥러닝(LSTM) 기반 나스닥(QQQ) 실전 매수 시그널")

y_pred_prob = ai_results['us_pred_prob']
y_test = ai_results['us_y_test']

y_pred_binary = (y_pred_prob > 0.5).astype(int)
hit_ratio = np.mean(y_pred_binary[:, 0] == y_test)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 6))
prob_1d = y_pred_prob.flatten()

ax.set_ylim(0.48, max(prob_1d) + 0.01)
ax.axhline(y=0.5, color='crimson', linestyle='--', label='기본 매수 기준점 (50%)', linewidth=2)

strong_buy_line = np.percentile(prob_1d, 75)
ax.axhline(y=strong_buy_line, color='goldenrod', linestyle=':', linewidth=2, label='강력 매수 기준선 (상위 25%)')

ax.plot(prob_1d, color='#1A5F7A', label='AI 예측 상승 확률', linewidth=2.5)

ax.fill_between(range(len(prob_1d)), 0.5, prob_1d,
                 where=(prob_1d > 0.5),
                 color='#22A699', alpha=0.25, label='일반 매수(Long) 유지 구간')

ax.fill_between(range(len(prob_1d)), strong_buy_line, prob_1d,
                 where=(prob_1d >= strong_buy_line),
                 color='#F2BE22', alpha=0.6, label='강력 매수 (Strong Buy) 구간')

ax.set_title(f'[한·미·중 동조화 AI 모델] 나스닥(QQQ) 방향성 예측 시그널 (적중률: {hit_ratio * 100:.1f}%)', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('시간 흐름 (최근 테스트 기간 300일)', fontsize=12)
ax.set_ylabel('AI 상승 확신도 (Probability)', fontsize=12)
ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')

st.pyplot(fig)
st.success("✅ 최신 시장 데이터를 반영한 한/미/중 AI 예측 및 시그널 업데이트가 완료되었습니다!")