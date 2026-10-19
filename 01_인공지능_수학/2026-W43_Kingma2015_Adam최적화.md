# Adam: 확률적 최적화 방법

> 원제: *Adam: A Method for Stochastic Optimization*

| 항목 | 내용 |
|---|---|
| 저자 | Diederik P. Kingma (University of Amsterdam), Jimmy Lei Ba (University of Toronto) |
| 학회 | ICLR 2015 |
| 분야 | 인공지능 수학 · 최적화 이론 |
| 난이도 | ★★★☆☆ |
| 링크 | https://arxiv.org/abs/1412.6980 |
| PDF 원문 | [arXiv PDF (무료)](https://arxiv.org/pdf/1412.6980) |
| ai_life 연결 단원 | 01 선형회귀(경사하강·학습률 슬라이더) |

---

## 한 줄 요약
**기울기의 1차 모멘트(평균)와 2차 모멘트(제곱 평균)를 지수이동평균으로 추정해, 파라미터마다 학습률을 자동으로 조절하는 최적화 알고리즘.**

## 왜 중요한가
오늘날 거의 모든 딥러닝 모델(GPT 포함)이 Adam 또는 그 변형(AdamW)으로 학습됩니다. 알고리즘은 몇 줄이지만, 각 줄에 **통계적 추정**과 **스케일 불변성**이라는 수학적 이유가 숨어 있습니다.

## 핵심 내용

### 알고리즘 (논문 Algorithm 1)
기울기 $g_t = \nabla_\theta f_t(\theta_{t-1})$ 에 대해
$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t &&\text{(1차 모멘트: 모멘텀)}\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 &&\text{(2차 모멘트: 크기)}\\
\hat m_t &= \frac{m_t}{1-\beta_1^t},\quad \hat v_t = \frac{v_t}{1-\beta_2^t} &&\text{(편향 보정)}\\
\theta_t &= \theta_{t-1} - \alpha\, \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\end{aligned}
$$
기본값: $\alpha=0.001,\ \beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$

### 각 단계의 수학적 의미
1. **편향 보정**: $m_0=0$으로 시작하므로 초기에는 0쪽으로 치우침. $\mathbb{E}[m_t] = \mathbb{E}[g](1-\beta_1^t)$ 이므로 $(1-\beta_1^t)$로 나누면 불편추정량이 됨. → **직접 유도해 보세요** (논문 3절).
2. **스케일 불변성**: 기울기를 $c$배 하면 $\hat m$도 $c$배, $\sqrt{\hat v}$도 $c$배 → 업데이트 크기는 그대로.
3. **유효 스텝 크기 상한**: 대략 $|\Delta\theta| \lesssim \alpha$ → $\alpha$가 "신뢰 영역(trust region)" 역할.
4. **신호 대 잡음비(SNR)**: $\hat m/\sqrt{\hat v}$는 기울기 방향이 얼마나 확실한지를 나타냄. 불확실하면 작게 움직임.

### 수렴 분석 (4절)
온라인 볼록 최적화 틀에서 regret $R(T) = O(\sqrt{T})$ 를 주장.
> ⚠️ 주의: 이 증명에는 오류가 있다는 것이 이후 밝혀짐 (Reddi et al., 2018, *On the Convergence of Adam and Beyond*). 논문을 비판적으로 읽는 좋은 연습 대상!

## 읽는 순서 가이드
1. **Algorithm 1** – 먼저 알고리즘을 손으로 써 보기 (15분)
2. **2절** – 각 하이퍼파라미터의 의미 (20분)
3. **3절 편향 보정** – 기댓값 계산 직접 따라가기 (30분)
4. **4절 수렴 분석** – 정리 진술만, 그리고 Reddi et al.의 반례와 비교 (20분)
5. **6절 실험** – 그래프만 훑기

## 사전 지식 체크
- [ ] 확률적 경사하강법(SGD)과 모멘텀
- [ ] 지수이동평균과 기댓값 계산
- [ ] 볼록함수와 regret 개념 (선택)

## 생각해볼 질문
1. $\beta_2$가 $\beta_1$보다 훨씬 1에 가까운 이유는 무엇일까?
2. Adam의 업데이트는 대각 헤시안 근사를 사용하는 2차 방법(뉴턴법)과 어떤 점이 닮았고 어떤 점이 다른가?
3. 가중치 감쇠(L2 정규화)를 Adam에 그대로 넣으면 왜 문제가 될까? (힌트: AdamW, Loshchilov & Hutter 2019)

## 용어 정리
| 영어 | 한국어 |
|---|---|
| first/second moment | 1차/2차 모멘트 |
| bias correction | 편향 보정 |
| exponential moving average | 지수이동평균 |
| regret bound | 후회 상한 |
| adaptive learning rate | 적응적 학습률 |

## 다음에 이어 읽을 논문
- Reddi, Kale, Kumar (2018), *On the Convergence of Adam and Beyond*
- Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization* (AdamW)
