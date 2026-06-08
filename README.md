# AI-Quant-Dashboard 📊

글로벌 거시경제 동조화(Coupling)를 반영한 **나스닥(QQQ) 방향성 예측 퀀트 모델**입니다.
미국, 한국, 중국 3개국 주요 지수 데이터를 수집/정제하고, LSTM 딥러닝 모델을 통해 내일의 시장 방향성을 예측하는 인터랙티브 대시보드를 구축하였습니다.

## 🚀 프로젝트 개요
* **목표:** 단일 국가 지수 분석의 한계를 넘어, 글로벌 증시 상관관계를 분석하여 나스닥 시장의 단기 방향성을 예측.
* **주요 기술:** Python, LSTM (Deep Learning), Streamlit (Web Dashboard), yfinance (Data API).
* **학습 데이터:** 2020년 1월 ~ 2026년 6월 (미국 QQQ, 한국 KODEX 200, 중국 ASHR).

## 🛠 Tech Stack
- **Language:** Python
- **Framework/Library:** - `TensorFlow/Keras` (LSTM 모델 구축)
  - `yfinance` (데이터 수집)
  - `Streamlit` (웹 시각화)
  - `Pandas/NumPy` (데이터 전처리)
  - `Matplotlib/Plotly` (데이터 시각화)

## 📌 주요 기능
1. **데이터 파이프라인:** `yfinance`를 활용한 3개국 실시간 데이터 수집 및 전처리 (휴장일 보정, 결측치 처리).
2. **AI 모델링:** 과거 10일간의 데이터를 입력값으로 사용하는 시계열(Time-series) LSTM 예측 모델.
3. **인터랙티브 대시보드:** Streamlit을 활용하여 실시간 예측 시그널 및 매수 구간(기본/강력)을 시각화.

## 📈 분석 결과
* **모델 성능:** 실전 테스트 데이터 기준 적중률 56.2% 달성.
* **시각화:** 상승 확신도(Probability)를 기반으로 한 직관적인 매수 신호 차트 제공.

## 🖥 프로젝트 화면
<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/1a9a685d-cd9b-4164-84a5-ca61596f706d" />
<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/9f5d6a25-b020-4262-b871-5d6e3e16c64f" />



## 🛠 설치 및 실행 방법
1. 저장소 복제:
   ```bash
   git clone [https://github.com/rhdqor1/ai-quant-dashboard.git](https://github.com/rhdqor1/ai-quant-dashboard.git)
2. 라이브러리 설치:
  pip install -r requirements.txt
3. 대시보드 실행:
   streamlit run app.py
