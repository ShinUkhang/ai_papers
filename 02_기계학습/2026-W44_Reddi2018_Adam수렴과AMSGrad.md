# Adam의 수렴에 대하여, 그리고 그 너머 (AMSGrad)

> 원제: *On the Convergence of Adam and Beyond*

| 항목 | 내용 |
|---|---|
| 저자 | Sashank J. Reddi, Satyen Kale, Sanjiv Kumar (Google 뉴욕) |
| 학회 | ICLR 2018 (최우수 논문상) |
| 분야 | 기계학습 · 최적화 이론 · 온라인 볼록 최적화 |
| 난이도 | ★★★★☆ |
| 링크 | https://openreview.net/forum?id=ryQu7f-RZ · https://arxiv.org/abs/1904.09237 |
| PDF 원문 | [arXiv PDF (무료)](https://arxiv.org/pdf/1904.09237) — 23쪽 (본문 8 + 참고문헌 1 + 부록 증명 14) |
| ai_life 연결 단원 | 01 선형회귀 심화(학습률), 03 CNN(학습에 쓰는 옵티마이저) |

---

## 한 줄 요약
**Adam·RMSProp은 과거 기울기 제곱의 '지수이동평균'으로 학습률을 조절하기 때문에, 드물게 오는 큰 기울기를 금방 잊어버려 아주 단순한 1차원 볼록 문제에서도 최악의 점으로 수렴할 수 있다. 과거 최댓값을 기억하는 AMSGrad로 고치면 수렴이 보장된다.**

## 왜 중요한가
지난주 Kingma & Ba(2015)가 Adam의 수렴을 "증명"했지만, 이 논문은 그 증명에 **오류**가 있음을 보이고 구체적인 **반례**를 만들었습니다. "널리 쓰이는 알고리즘도 수학적으로 틀릴 수 있다", "이론적 증명은 꼼꼼히 검증해야 한다"는 교훈을 준 사건이자, 최적화 알고리즘 설계에서 **학습률이 증가하면 안 된다**(Γ ≽ 0)는 원리를 명확히 했습니다. ICLR 2018 최우수 논문상.

## 핵심 내용

### 1. 일반 적응형 방법의 틀 (Algorithm 1)
$$\mathbf x_{t+1} = \Pi_{\mathcal F,\sqrt{V_t}}\!\left(\mathbf x_t - \alpha_t\, \mathbf m_t/\sqrt{V_t}\right),\quad \mathbf m_t=\phi_t(\mathbf g_1..\mathbf g_t),\ V_t=\psi_t(\mathbf g_1..\mathbf g_t)$$
| 방법 | $\phi_t$ (1차) | $\psi_t$ (2차) |
|---|---|---|
| SGD | $\mathbf g_t$ | $I$ |
| AdaGrad | $\mathbf g_t$ | $\mathrm{diag}(\sum_i \mathbf g_i^2)/t$ — **모든** 과거 |
| Adam | $(1-\beta_1)\sum\beta_1^{t-i}\mathbf g_i$ | $(1-\beta_2)\mathrm{diag}(\sum\beta_2^{t-i}\mathbf g_i^2)$ — **최근** 위주 |

### 2. 문제의 핵심량
$$\Gamma_{t+1} = \frac{\sqrt{V_{t+1}}}{\alpha_{t+1}} - \frac{\sqrt{V_t}}{\alpha_t}$$
= "학습률의 역수"의 변화. SGD·AdaGrad는 $\Gamma_t\succeq0$ (학습률이 **증가하지 않음**). Adam은 $\Gamma_t$가 음수가 될 수 있다 — 원래 Adam 증명은 이를 양반정치로 **잘못 가정**.

### 3. 반례 (정리 1) ⭐
$\mathcal F=[-1,1]$, $C>2$:
$$f_t(x)=\begin{cases} Cx, & t \bmod 3 = 1\\ -x, & \text{그 외}\end{cases}$$
세 번에 한 번 큰 기울기 $C$(올바른 방향 $x\to-1$), 두 번은 $-1$(잘못된 방향). 최적해는 $x=-1$. 하지만 $\beta_1=0$, $\beta_2=1/(1+C^2)$인 Adam은 큰 기울기를 거의 $C$배로 **눌러 버려서** $x=+1$(최악의 점)로 간다. 3스텝마다 regret $\ge 2C-4$ → 평균 regret이 0으로 가지 않음.
- 정리 2: $\beta_1<\sqrt{\beta_2}$인 **어떤** 상수 $\beta_1,\beta_2$에도 반례 존재
- 정리 3: 확률적(stochastic) 설정에서도 반례
- 정리 6: 분모에 $\epsilon$을 더해도 소용없음

### 4. 해결책: AMSGrad (Algorithm 2)
$$\hat v_t=\max(\hat v_{t-1}, v_t),\qquad \mathbf x_{t+1}=\Pi\big(\mathbf x_t-\alpha_t\,\mathbf m_t/\sqrt{\hat v_t}\big)$$
과거 $v$의 **최댓값**을 기억 → 학습률이 커지지 않음($\Gamma_t\succeq0$), "장기 기억". 메모리·시간은 Adam과 같음. 정리 4: regret $O(\sqrt T)$ (데이터 의존적으로는 AdaGrad처럼 더 좋을 수 있음).

### 5. 온라인 최적화와 regret
$$R_T=\sum_{t=1}^T f_t(\mathbf x_t)-\min_{\mathbf x\in\mathcal F}\sum_{t=1}^T f_t(\mathbf x)$$
$R_T=o(T)$ ⇔ 평균 성능이 최적에 수렴. 온라인 경사하강의 기본 학습률은 $\alpha/\sqrt t$.

### 6. 실험
합성 문제(1010x / −10x)에서 Adam은 $x=1$로, AMSGrad는 $x=-1$로. MNIST 로지스틱 회귀·1층 신경망(ReLU 100개), CIFAR-10(CIFARNET)에서 AMSGrad가 같거나 더 좋음.

## 읽는 순서 가이드
1. **p.1–2 초록·서론·기여** – "드물지만 큰 기울기를 빨리 잊는다" (15분)
2. **p.2–3 Algorithm 1과 표** – SGD/AdaGrad/Adam을 한 틀로 (30분)
3. **p.4 Γ와 반례** – 반례를 손으로 몇 스텝 시뮬레이션 ⭐ (40분)
4. **p.5 AMSGrad** – max 한 줄의 의미 (15분)
5. **p.6–8 정리 4·실험·결론** (20분)
6. **p.10–11 정리 1 증명** – 귀납법 구조만 (선택, 30분)

## 사전 지식 체크
- [ ] Kingma & Ba(2015) Adam (지난주)
- [ ] 이상구 외(2023) 경사하강법·SGD (이번 주)
- [ ] 지수이동평균 $m_t=\beta m_{t-1}+(1-\beta)g_t$
- [ ] 볼록함수, 사영(projection)
- [ ] regret, $O(\sqrt T)$ 표기

## 생각해볼 질문
1. 반례에서 $C=10$이면 $\beta_2=1/101$. 이 $\beta_2$는 실전 값 0.999와 매우 다르다. 그런데 정리 2는 왜 실전 값에서도 문제가 생길 수 있다고 말하는가?
2. AMSGrad의 "max"는 학습률을 **줄이기만** 한다. 그렇다면 왜 AdaGrad처럼 학습률이 너무 빨리 줄지 않는가?
3. 이 논문 이후 실무에선 여전히 Adam(AdamW)을 많이 쓴다. 이론과 실전 사이의 간극을 어떻게 이해할까?
4. "드물지만 정보가 많은 큰 기울기"는 실제 데이터의 어떤 상황에 해당할까? (힌트: 출력 공간이 큰 문제, 희귀 클래스)

## 용어 정리
| 영어 | 한국어 |
|---|---|
| adaptive method | 적응형 방법 |
| exponential moving average (EMA) | 지수이동평균 |
| online convex optimization | 온라인 볼록 최적화 |
| regret | 후회(누적 손실 차이) |
| projection | 사영 |
| positive semi-definite | 양반정치 |
| long-term memory | 장기 기억 |
| counterexample | 반례 |

## 다음에 이어 읽을 논문
- Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization* (AdamW, ICLR 2019)
- Duchi, Hazan, Singer (2011), *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization* (AdaGrad, JMLR)
