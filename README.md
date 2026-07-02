## 1. 담당 범위

```
Simulation Agent 파이프라인
1단계: 데이터 수집    (data_collector.py)   - Data Agent DB에서 주가/매크로 수집
2단계: 전처리         (preprocessor.py)     - 로그수익률 계산, LSTM 텐서 구성
3단계: 리스크 분류    (risk_classifier.py)  - LLM으로 매크로 리스크 요인 분류
4단계: LSTM 예측      (model.py)            - 수익률 분포(mu, sigma) 예측 + What-if
5단계: Monte Carlo    (monte_carlo.py)      - 1,000개 가격 경로 시뮬레이션
6단계: 결과 패키징    (result_packager.py)  - 리스크 프로필 기반 해석 문구 생성
7단계: 프론트 전송    (service.py)          - 최종 결과 전송
```

---

## 2. 사용 기술

- **LLM**: Upstage `solar-pro` (리스크 요인 분류)
- **모델**: PyTorch LSTM (수익률 분포 학습)
- **시뮬레이션**: Monte Carlo (1,000-path, 30일)
- **DB**: Data Agent의 SQLite `reports.db` (주가/매크로 데이터)

---

## 3. 지원 종목

| 종목코드 | 기업명 |
|---|---|
| 005930 | 삼성전자 |
| 000660 | SK하이닉스 |
| 005380 | 현대차 |
| 035420 | NAVER |
| 003230 | 삼양식품 |
| 352820 | HYBE |
| 373220 | LG에너지솔루션 |

---

## 4. 파일 구조

```
financial_research_simulation_agent/
├── requirements.txt
├── .env.example
│
└── simulation_agent/
    ├── service.py              # 진입점: run_simulation()
    ├── data_collector.py       # Data Agent DB에서 데이터 수집
    ├── preprocessor.py         # 피처 테이블 생성 (로그수익률, 변동성 등)
    ├── risk_classifier.py      # LLM 기반 매크로 리스크 요인 분류
    ├── model.py                # LSTM 학습/예측/캐싱, apply_shock()
    ├── monte_carlo.py          # Monte Carlo 경로 생성 및 시나리오 요약
    ├── result_packager.py      # 결과 패키징 및 해석 문구 생성
    └── _external_deps.py       # sys.path 연결
```
---

## 5. 사전 준비

이 코드는 아래 두 리포가 먼저 실행되어 있어야 합니다.

- (https://github.com/boogiewooki02/financial-research-agent): 주가/매크로 데이터 수집 → `reports.db` 생성
- (https://github.com/nasuzz-dev/financial_research_data_agent): `get_agent_context()` 등 공통 함수 제공

---

## 6. 환경 설정

가상환경 설치:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`.env` 파일 생성 (루트 폴더에):

```
# Data Agent 리포 경로
DATA_AGENT_REPO_PATH=../financial_research_data_agent

# Data Agent DB 경로
B_DB_PATH=../financial-research-agent/db/reports.db

# Upstage API 키 (리스크 분류 LLM)
UPSTAGE_API_KEY=your_upstage_key
```

---

## 7. 실행 방법

**전체 파이프라인 실행 (삼성전자)**

```powershell
python -m simulation_agent.service --ticker 005930
```

**다른 종목 실행**

```powershell
python -m simulation_agent.service --ticker 000660
```

**단계별 개별 테스트**

```powershell
# LSTM 모델 학습/예측 테스트
python -m simulation_agent.model --ticker 005930

# Monte Carlo 테스트
python -m simulation_agent.monte_carlo --ticker 005930

# 결과 패키징 테스트
python -m simulation_agent.result_packager --ticker 005930
```

---

## 8. Debate Agent 연동 방법

Debate Agent는 아래 함수를 호출하면 됩니다.

```python
from simulation_agent.service import run_simulation

await run_simulation(
    ticker="005930",
    user_id="u1",
    agenda_2={
        "bull_summary": "환율 상승이 수출에 긍정적",
        "bull_arguments": "원달러 환율 상승으로 수출 채산성 개선",
        "bear_summary": "금리 인상 우려",
        "bear_arguments": "기준금리 추가 인상 시 투자심리 위축 우려",
    },
)
```

**결과 예시**

```json
{
  "ticker": "005930",
  "current_price": 286000.0,
  "simulation_type": "what_if",
  "summary": {
    "expected_return_pct": 23.24,
    "upside_probability": 89.4,
    "volatility": 19.79
  },
  "scenarios": {
    "optimistic": {
      "expected_return_pct": 62.41,
      "avg_final_price": 464491.0,
      "path_count": 100
    },
    "neutral": {
      "expected_return_pct": 22.10,
      "avg_final_price": 349210.0,
      "path_count": 800
    },
    "pessimistic": {
      "expected_return_pct": -6.85,
      "avg_final_price": 266406.0,
      "path_count": 100
    }
  },
  "interpretation": "향후 30일 예상 평균 수익률은 23.2%이며 ...",
  "risk_card": {
    "volatility": 19.79,
    "pessimistic_return_pct": -6.85,
    "quantified_risks": [],
    "qualitative_risks": [],
    "tone": "moderate"
  },
  "user_profile": {
    "risk_profile": "moderate",
    "investment_goal": "mid_term"
  }
}
```

---

## 9. 모델 캐싱

학습된 LSTM 모델은 `simulation_agent/_model_cache/{ticker}.pt`에 저장됩니다.

- 캐시가 **24시간 이내**이면 재사용
- **24시간 초과** 시 자동으로 재학습
- 강제 재학습: `get_or_train_model(..., force_retrain=True)`

---

## 10. 현재 상태 및 제한사항

- `get_user_context()`는 아직 목업 데이터 반환 중 (실제 유저 DB 연결 필요)
- `send_result_fn`이 None이면 로그로만 출력 (WebSocket/HTTP 연결 시 교체)
- What-if 시나리오는 `agenda_2`가 실제 연결되어야 정상 동작
- 표준화는 윈도우 단위로 처리 중 (향후 전체 기준 scaler로 개선 가능)
