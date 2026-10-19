# p.1 해설 — 초록과 서론

**① 초록** — Adam은 **확률적 목적함수**를 위한 1차(기울기만 쓰는) 최적화 알고리즘. 기울기의 **저차 모멘트를 적응적으로 추정**한다. 장점 목록:
- 구현이 쉽고 계산 효율적, 메모리 적음
- 기울기의 **대각 스케일 변환에 불변**
- 데이터·파라미터가 큰 문제, **비정상(non-stationary)** 목적함수, 잡음 많거나 **희소한** 기울기에 적합
- 하이퍼파라미터가 직관적이고 조정이 거의 필요 없음
- 온라인 볼록 최적화에서 최선 수준의 **regret 상한** 증명, 무한대 노름 변형 **AdaMax**도 제시
→ 🔎 *invariant to diagonal rescaling of the gradients* 에 밑줄. p.3에서 증명됨.

**② 서론 첫 문단** — 많은 과학·공학 문제는 **스칼라 목적함수의 최적화**. 1차 편미분 계산 비용은 함수값 계산과 같은 수준이라 경사하강법이 효율적. 목적함수가 확률적인 경우(미니배치, 드롭아웃 잡음) → **SGD**. 고차원에서는 2차 방법이 부적합하므로 1차 방법만 다룬다.

**③ 둘째 문단 — Adam의 정체** — 파라미터마다 **개별 적응 학습률**을 기울기의 1차·2차 모멘트 추정으로 계산. 이름 = **ada**ptive **m**oment estimation. 두 방법의 장점 결합:
- **AdaGrad**: 희소 기울기에 강함
- **RMSProp**: 온라인·비정상 환경에 강함

**④ 각주** — 두 저자의 공동 1저자 순서는 **구글 행아웃에서 동전 던지기로** 정했다는 재미있는 각주.

<div style="page-break-after: always"></div>

# p.2 해설 — Algorithm 1과 2절 알고리즘

**① Algorithm 1 ⭐ 손으로 옮겨 쓰기** — 기본값 $\alpha=0.001,\ \beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$. 모든 연산은 **원소별**.
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat m_t = \frac{m_t}{1-\beta_1^t},\quad \hat v_t = \frac{v_t}{1-\beta_2^t},\quad \theta_t = \theta_{t-1} - \alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$

**② 2절 설정** — $f(\theta)$는 잡음 있는 목적함수, 목표는 기댓값 $\mathbb{E}[f(\theta)]$ 최소화. $f_1, \dots, f_T$는 시점마다의 실현값(미니배치 등).

**③ 이동평균의 의미** — $m_t$는 기울기의 **1차 모멘트(평균)**, $v_t$는 **2차 원 모멘트(중심화하지 않은 분산)** 의 추정. 0으로 초기화하므로 초기에는 **0쪽으로 치우침**, 특히 $\beta$가 1에 가까울 때. → 3절에서 보정.

**④ 효율적인 계산 순서** — 마지막 세 줄을 다음으로 대체 가능:
$$\alpha_t = \alpha\cdot\frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t},\qquad \theta_t = \theta_{t-1} - \alpha_t\frac{m_t}{\sqrt{v_t}+\hat\epsilon}$$

**⑤ 2.1 업데이트 규칙** — $\epsilon=0$일 때 유효 스텝 $\Delta_t = \alpha\,\hat m_t/\sqrt{\hat v_t}$. 상한 두 가지: $(1-\beta_1) > \sqrt{1-\beta_2}$ 이면 $|\Delta_t| \le \alpha(1-\beta_1)/\sqrt{1-\beta_2}$, 아니면 $|\Delta_t| \le \alpha$.

<div style="page-break-after: always"></div>

# p.3 해설 — 신뢰 영역, SNR, 3절 편향 보정

**① 스텝 크기 상한의 의미** — 첫 번째 경우는 "현재 시점만 빼고 기울기가 늘 0"인 극단적 희소 상황에서만 생긴다. 보통은 $|\mathbb{E}[g]|/\sqrt{\mathbb{E}[g^2]} \le 1$ 이므로 $|\Delta_t| \lessapprox \alpha$.
→ 🔎 *trust region* 에 밑줄. $\alpha$는 "현재 기울기를 믿을 수 있는 반경".

**② $\alpha$를 미리 정하기 쉬움** — 좋은 최적점이 대략 어느 영역에 있는지 알면, 몇 번의 반복 안에 도달할 수 있는 $\alpha$의 크기를 추론할 수 있다.

**③ 신호 대 잡음비(SNR)** — $\hat m_t/\sqrt{\hat v_t}$. SNR이 작으면 = 방향이 불확실 → 스텝이 작아짐. 최적점 근처에서 SNR → 0 → **자동 담금질(annealing)**.

**④ 스케일 불변성 증명 (한 줄!)** — 기울기를 $c$배: $\dfrac{c\,\hat m_t}{\sqrt{c^2 \hat v_t}} = \dfrac{\hat m_t}{\sqrt{\hat v_t}}$.

**⑤ 3절: 편향 보정 유도 ⭐ 직접 따라가기**
$$v_t = (1-\beta_2)\sum_{i=1}^{t}\beta_2^{t-i} g_i^2 \tag{1}$$
양변의 기댓값:
$$\mathbb{E}[v_t] = \mathbb{E}[g_t^2]\,(1-\beta_2)\sum_{i=1}^{t}\beta_2^{t-i} + \zeta = \mathbb{E}[g_t^2]\,(1-\beta_2^t) + \zeta$$
등비급수 $\sum_{i=1}^t \beta_2^{t-i} = \frac{1-\beta_2^t}{1-\beta_2}$ 를 쓰면 됨. 2차 모멘트가 정상(stationary)이면 $\zeta=0$. → **$(1-\beta_2^t)$로 나누면 불편추정**.

**⑥ 왜 중요한가** — 희소 기울기에서는 $\beta_2$를 1에 가깝게 해야 하는데, 바로 그때 보정이 없으면 **초기 스텝이 매우 커진다**.

<div style="page-break-after: always"></div>

# p.4 해설 — 4절 수렴 분석

**① 온라인 학습 틀 (Zinkevich 2003)** — 미지의 볼록 비용함수 열 $f_1, \dots, f_T$. 매 시점 $\theta_t$를 고른 뒤 $f_t$로 평가. 성능 척도 **regret**:
$$R(T) = \sum_{t=1}^{T}\big[f_t(\theta_t) - f_t(\theta^*)\big],\qquad \theta^* = \arg\min_{\theta\in\mathcal X}\sum_t f_t(\theta) \tag{5}$$
→ "처음부터 최선의 고정 답을 알았다면"과 비교한 누적 손해.

**② 결과** — Adam은 $R(T) = O(\sqrt T)$. 일반 볼록 온라인 학습의 최선 결과와 동급.

**③ 정리 4.1 가정** — 기울기 유계($\|\nabla f_t\|_2\le G$, $\|\cdot\|_\infty\le G_\infty$), 파라미터 거리 유계($D$, $D_\infty$), $\beta_1^2/\sqrt{\beta_2}<1$, 학습률 $\alpha_t = \alpha/\sqrt t$, $\beta_{1,t} = \beta_1\lambda^{t-1}$ (지수적으로 감소).

**④ 해석** — 데이터 특징이 희소하면 상한의 합이 $dG_\infty\sqrt T$ 보다 **훨씬 작을 수** 있다. 적응적 방법(Adam, AdaGrad)은 $O(\log d\,\sqrt T)$, 비적응적 방법은 $O(\sqrt{dT})$.

**⑤ 따름정리 4.2** — 평균 regret $R(T)/T = O(1/\sqrt T) \to 0$.

> ⚠️ **비판적 읽기**: 이 증명에는 오류가 있음이 2018년 밝혀졌다 (Reddi 외, *On the Convergence of Adam and Beyond*). 어느 단계가 문제인지는 부록(p.12–15)을 읽으며 찾아보세요.

**⑥ 5절 관련 연구 시작** — RMSProp, AdaGrad가 직접 관련. vSGD, AdaDelta, natural Newton 등은 1차 정보로 곡률 추정.

<div style="page-break-after: always"></div>

# p.5 해설 — 관련 연구: 자연 경사, RMSProp, AdaGrad

**① 자연 경사(NGD)와의 관계 ⭐ 수학적으로 중요** — $\hat v_t$는 **피셔 정보 행렬의 대각 성분**의 근사. Adam은 데이터의 기하에 맞춘 **사전조건자(preconditioner)** 를 쓰지만, NGD보다 보수적으로 **역 대각 피셔 행렬의 제곱근**을 쓴다.
→ 🔎 *diagonal of the Fisher information matrix* 에 밑줄. 정보기하학(Amari)과 연결.

**② RMSProp과의 차이**
- RMSProp+모멘텀: **재조정된 기울기**에 모멘텀 적용
- Adam: 1·2차 모멘트의 이동평균으로 **직접** 추정
- RMSProp에는 **편향 보정이 없음** → $\beta_2 \approx 1$에서 큰 스텝·발산 (6.4절에서 실험)

**③ AdaGrad와의 대응 ⭐ 유도해 보기** — AdaGrad: $\theta_{t+1} = \theta_t - \alpha\, g_t / \sqrt{\sum_{i=1}^t g_i^2}$.
Adam에서 $\beta_1 = 0$, $\beta_2 \to 1$, $\alpha_t = \alpha t^{-1/2}$ 로 두면
$$\lim_{\beta_2\to1}\hat v_t = \frac1t\sum_{i=1}^t g_i^2 \;\Rightarrow\; \theta_t - \alpha t^{-1/2}\frac{g_t}{\sqrt{t^{-1}\sum g_i^2}} = \theta_t - \alpha\frac{g_t}{\sqrt{\sum g_i^2}}$$
→ AdaGrad는 Adam의 **특수한 경우**. 단, 편향 보정이 있어야만 이 대응이 성립.

**④ 6절 실험 시작** — 로지스틱 회귀, 다층 신경망, CNN. 같은 초기화, 하이퍼파라미터는 격자 탐색 후 최선값.

**⑤ 6.1 로지스틱 회귀** — 볼록 문제라 국소 최솟값 걱정 없이 비교 가능. MNIST(784차원), $\alpha_t = \alpha/\sqrt t$. IMDB 영화 리뷰 1만 단어 BoW(매우 희소) + 50% 드롭아웃.

<div style="page-break-after: always"></div>

# p.6 해설 — 실험 결과: 로지스틱 회귀, 다층 신경망

**① Figure 1** — 왼쪽(MNIST): Adam ≈ SGD+Nesterov, 둘 다 AdaGrad보다 빠름. 오른쪽(IMDB 희소 특징): AdaGrad가 SGD보다 크게 앞서고, **Adam은 AdaGrad만큼 빠름**.
→ 희소 특징에서 적응적 학습률의 힘. p.5의 이론(AdaGrad 대응)과 일치.

**② 6.2 다층 신경망** — **비볼록** 문제라 수렴 이론은 적용되지 않지만, 실험적으로 Adam이 자주 더 좋다. 구조: 은닉층 2개 × 1000개, ReLU, 미니배치 128.

**③ SFO와 비교** — SFO(준뉴턴 방법)는 곡률 갱신 비용 때문에 반복당 **5~10배 느리고** 메모리가 미니배치 수에 비례. 확률적 정규화(드롭아웃)가 있으면 **수렴 실패**.

**④ 6.3 CNN 시작** — CNN은 가중치 공유 때문에 **층마다 기울기 크기가 매우 다르다** → SGD에서는 합성곱 층에 작은 학습률을 따로 주곤 함. 구조: 5×5 합성곱 + 3×3 최대 풀링 3단계 + 완전연결 1000개.
→ 🔎 *vastly different gradients in different layers* — 파라미터별 적응 학습률이 필요한 이유.

<div style="page-break-after: always"></div>

# p.7 해설 — Figure 2, 3: 신경망과 CNN 결과

**① Figure 2** — (a) 드롭아웃 있는 MLP: Adam이 AdaGrad, RMSProp, SGD+Nesterov, AdaDelta보다 빠르게 수렴. (b) 결정적 비용함수: SFO와 비교.

**② Figure 3 (CIFAR-10 CNN)** — 왼쪽(처음 3 에폭): Adam과 AdaGrad 모두 초기에 빠르게 감소. 오른쪽(45 에폭): **Adam과 SGD가 AdaGrad보다 훨씬 빠르게** 수렴.

**③ 흥미로운 관찰 ⭐** — CNN에서는 몇 에폭 후 $\hat v_t$가 거의 0이 되어 **$\epsilon$이 지배**한다. 즉 2차 모멘트가 비용함수의 기하를 잘 반영하지 못함. 대신 **1차 모멘트로 미니배치 분산을 줄이는 것**이 속도 향상의 주원인.
→ 🔎 *dominated by the ε* 에 밑줄. 수학엔 없지만 **계산에서 필요한 $\epsilon$** 이 실제로 결과를 좌우하는 사례.

**④ 결론** — Adam의 SGD 대비 개선은 작지만, **층마다 학습률을 수동으로 고를 필요가 없다**는 것이 실용적 장점.

<div style="page-break-after: always"></div>

# p.8 해설 — 6.4 편향 보정 실험, 7.1 AdaMax 시작

**① Figure 4 ⭐** — VAE 학습에서 편향 보정 **있음(빨강) vs 없음(초록)**. x축 $\log_{10}\alpha$, 행 $\beta_1 \in \{0, 0.9\}$, 열 $\beta_2 \in \{0.99, 0.999, 0.9999\}$. $\beta_2$가 1에 가까울수록 보정 없을 때 **불안정**.
→ 이번 주 함께 읽는 **VAE 논문**의 모델로 실험! 같은 저자(Kingma).

**② 6.4 해석** — 편향 보정을 빼면 = **RMSProp + 모멘텀**. $\beta_2 \approx 1$ (희소 기울기에 필요)일수록 초기 편향이 커서 보정이 중요. 학습 후반에 은닉 유닛이 특정 패턴에 특화되면서 기울기가 **더 희소**해지므로 차이가 더 뚜렷. 결론: Adam ≥ RMSProp (모든 설정에서).

**③ 7.1 AdaMax: $L^2 \to L^p$ 일반화** — Adam은 기울기를 과거 기울기의 (스케일된) $L^2$ 노름에 반비례하게 조정. 이를 $L^p$ 노름으로 일반화:
$$v_t = \beta_2^p v_{t-1} + (1-\beta_2^p)|g_t|^p = (1-\beta_2^p)\sum_{i=1}^t \beta_2^{p(t-i)}|g_i|^p$$
$p$가 크면 수치적으로 불안정하지만, $p\to\infty$ 에서는 놀랍게도 **단순하고 안정적**인 알고리즘이 나온다.

<div style="page-break-after: always"></div>

# p.9 해설 — AdaMax 유도, 시간 평균, 결론

**① Algorithm 2 (AdaMax)** — 기본값 $\alpha=0.002$. 핵심 한 줄:
$$u_t = \max(\beta_2 u_{t-1},\ |g_t|),\qquad \theta_t = \theta_{t-1} - \frac{\alpha}{1-\beta_1^t}\frac{m_t}{u_t}$$

**② 극한 유도 ⭐ 해석학 연습** — $u_t = \lim_{p\to\infty} v_t^{1/p}$:
$$\lim_{p\to\infty}(1-\beta_2^p)^{1/p} = 1,\qquad \lim_{p\to\infty}\Big(\sum_i a_i^p\Big)^{1/p} = \max_i a_i$$
두 사실을 쓰면
$$u_t = \max\big(\beta_2^{t-1}|g_1|,\ \beta_2^{t-2}|g_2|,\ \dots,\ |g_t|\big)$$
→ ✍️ 두 극한을 직접 증명해 보세요. ($L^p$ 노름 → $L^\infty$ 노름)

**③ AdaMax의 장점** — $u_t$에는 **초기화 편향 보정이 필요 없고**, $|\Delta_t| \le \alpha$ 라는 더 단순한 상한.

**④ 7.2 시간 평균** — 마지막 반복값은 잡음이 많으므로 **평균**이 일반화에 유리. Polyak–Ruppert 평균 대신 파라미터의 **지수이동평균** $\bar\theta_t = \beta_2\bar\theta_{t-1} + (1-\beta_2)\theta_t$ 을 쓰고, 마찬가지로 $(1-\beta_2^t)$로 편향 보정.

**⑤ 8절 결론 시작** — 확률적 목적함수를 위한 단순하고 효율적인 알고리즘.

<div style="page-break-after: always"></div>

# p.10 해설 — 결론, 감사의 글, 참고문헌

**① 결론** — 대규모 데이터·고차원 파라미터 문제를 겨냥. **AdaGrad의 희소 기울기 처리 + RMSProp의 비정상 목적함수 처리**를 결합. 구현이 쉽고 메모리가 적음. 볼록 문제에서 수렴 속도 분석을 실험으로 확인했고, 다양한 **비볼록** 기계학습 문제에서 견고했다.

**② 감사의 글** — 이름 "Adam"은 Ivo Danihelka와 Tom Schaul이 지었다. Duke 대학의 Kai Fan이 **원래 AdaMax 유도의 오류**를 찾아줬다.
→ 🔎 논문도 버전이 갱신되며 오류가 고쳐진다는 것 (이 PDF는 arXiv **v9**, 2017년).

**③ 참고문헌 중 주목할 것**

| 참고문헌 | 왜 볼까 |
|---|---|
| Amari (1998) 자연 경사 | Adam의 정보기하학적 해석의 뿌리 |
| Duchi 외 (2011) AdaGrad | 적응적 학습률과 regret 분석 |
| Kingma & Welling (2013) VAE | 이번 주 함께 읽는 논문 |
| Moulines & Bach (2011) | 확률적 근사의 비점근 분석 |
| Polyak & Juditsky (1992) | 평균화로 SGD 가속 |

<div style="page-break-after: always"></div>

# p.11 해설 — 참고문헌 (계속)

**참고문헌 중 주목할 것**

| 참고문헌 | 왜 볼까 |
|---|---|
| Ruppert (1988) | Robbins–Monro 확률적 근사의 효율적 추정 (평균화) |
| Schaul, Zhang, LeCun (2012) *No more pesky learning rates* | 학습률 자동 조정(vSGD) |
| Sutskever 외 (2013) | 초기화와 모멘텀의 중요성 — 학습 후반 모멘텀 감소가 수렴을 돕는다 (p.4 $\beta_{1,t}$ 감소와 연결) |
| Tieleman & Hinton (2012) RMSProp | **논문이 아니라 Coursera 강의 슬라이드!** |
| Zeiler (2012) AdaDelta | 학습률 없는 적응적 방법 |
| Zinkevich (2003) | 온라인 볼록 최적화와 regret — 4절의 틀 |

→ 🔎 RMSProp이 정식 논문 없이 **강의 자료**로만 발표되었는데도 널리 쓰인다는 점이 흥미롭습니다.

**✍️ 여기까지 정리 질문**
1. Adam의 세 가지 핵심 장치(모멘텀, 적응적 크기 조정, 편향 보정)를 각각 한 문장으로 설명해 보자.
2. AdaGrad, RMSProp, Adam, AdaMax의 관계를 그림으로 그려 보자.

<div style="page-break-after: always"></div>

# p.12 해설 — 부록 10.1: 수렴 증명 (1)

**① 정의 10.1 (볼록 함수)** — $f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y)$.

**② 보조정리 10.2 ⭐ 볼록 함수의 1차 조건**
$$f(y) \ge f(x) + \nabla f(x)^\top (y-x)$$
→ 볼록 함수는 **접평면 위에** 있다. regret을 기울기로 상한하는 출발점: $f_t(\theta_t) - f_t(\theta^*) \le g_t^\top(\theta_t - \theta^*)$.

**③ 보조정리 10.3** — 유계 기울기에 대해
$$\sum_{t=1}^T \sqrt{\frac{g_{t,i}^2}{t}} \le 2G_\infty \|g_{1:T,i}\|_2$$
**수학적 귀납법**으로 증명.
→ ✍️ 귀납 단계에서 $\sqrt{a - b} \le \sqrt a - \frac{b}{2\sqrt a}$ 꼴의 부등식이 쓰입니다. 직접 확인해 보세요.

**④ 읽기 팁** — 부록은 증명을 따라가는 연습장입니다. 한 줄씩 **왜 부등호가 성립하는지** 옆에 적어 가며 읽으세요.

<div style="page-break-after: always"></div>

# p.13 해설 — 부록: 보조정리 10.4, 정리 10.5

**① 보조정리 10.4** — $\gamma = \beta_1^2/\sqrt{\beta_2} < 1$ 이라 하면
$$\sum_{t=1}^T \frac{\hat m_{t,i}^2}{\sqrt{t\,\hat v_{t,i}}} \le \frac{2}{1-\gamma}\frac{1}{\sqrt{1-\beta_2}}\|g_{1:T,i}\|_2$$
증명: $\hat m_t$를 과거 기울기의 가중합으로 펼치고, 코시–슈바르츠 부등식과 등비급수 $\sum \gamma^k \le 1/(1-\gamma)$, 그리고 보조정리 10.3을 적용.
→ 🔎 조건 $\beta_1^2 < \sqrt{\beta_2}$ 가 어디서 쓰이는지 표시하세요.

**② 정리 10.5 (= 본문 정리 4.1)** — 가정과 결론을 다시 진술. 세 항의 합으로 regret 상한:
- 첫째 항: 파라미터 거리 $D$와 $\sqrt{T\hat v_{T,i}}$
- 둘째 항: 기울기 합 $\|g_{1:T,i}\|_2$
- 셋째 항: $\beta_{1,t}$ 감소율 $\lambda$에 의존

> ⚠️ Reddi 외(2018)가 지적한 문제: 증명은 $\sqrt{\hat v_t}/\alpha_t$ 가 **단조 증가**한다고 암묵적으로 가정하는데, Adam에서는 이것이 보장되지 않는다. 이 때문에 볼록 문제에서도 수렴하지 않는 반례가 존재 → **AMSGrad** 제안.

<div style="page-break-after: always"></div>

# p.14 해설 — 부록: 정리의 증명

**① 출발** — 보조정리 10.2(볼록성)로
$$f_t(\theta_t) - f_t(\theta^*) \le g_t^\top(\theta_t - \theta^*) = \sum_{i=1}^d g_{t,i}(\theta_{t,i} - \theta^*_{,i})$$

**② 업데이트 규칙 대입** — $\theta_{t+1} = \theta_t - \alpha_t \hat m_t/\sqrt{\hat v_t}$ 를 $(\theta_{t+1,i}-\theta^*_{,i})^2$ 에 넣고 전개하여 $g_{t,i}(\theta_{t,i}-\theta^*_{,i})$ 를 다른 항들로 표현.

**③ 영의 부등식(Young's inequality)** — $ab \le \frac{a^2}{2} + \frac{b^2}{2}$ 꼴을 써서 교차항을 정리.

**④ 망원 합(telescoping sum)** — $t=1$부터 $T$까지 더하면 $(\theta_t - \theta^*)^2$ 항들이 **연쇄적으로 상쇄**. 이때 앞서 말한 단조성 가정이 들어간다.
→ 🔎 망원 합이 나오는 줄에 표시하고, 상쇄가 정말 일어나는지 직접 확인해 보세요. **이 줄이 Reddi 외가 지적한 곳입니다.**

**⑤ 보조정리 10.4 적용** — 남은 $\hat m^2/\sqrt{\hat v}$ 합을 기울기 노름으로 상한.

<div style="page-break-after: always"></div>

# p.15 해설 — 증명 마무리와 전체 정리

**① 마무리** — $\beta_{1,t} = \beta_1\lambda^{t-1}$ 감소에서 생기는 항을 등비급수로 상한 → $\frac{1}{(1-\lambda)^2}$ 꼴. 세 항을 합쳐 정리 10.5 완성.

**② 결론의 의미** — 가정이 맞다면 $R(T) = O(\sqrt T)$, 평균 regret → 0.

**✍️ 논문 전체 정리 질문**
1. 편향 보정 계수 $(1-\beta^t)$를 등비급수로 직접 유도할 수 있는가? (p.3)
2. Adam의 $\hat v_t$가 피셔 정보 행렬의 대각 근사라는 말은 무슨 뜻인가? 자연 경사와 비교해 보자. (p.5)
3. AdaMax 유도에서 $L^p \to L^\infty$ 극한 두 개를 증명해 보자. (p.9)
4. 이 증명의 어느 단계가 틀렸는지 Reddi 외(2018)를 읽고 찾아보자. (p.14)
5. ai_life **01 선형회귀** 시뮬레이터의 학습률 슬라이더에 Adam을 적용하면 무엇이 달라질까?

| 핵심 용어 | 뜻 |
|---|---|
| regret | 후회 (최선의 고정 답과의 누적 차이) |
| preconditioner | 사전조건자 |
| telescoping sum | 망원 합 |
| Fisher information | 피셔 정보 |
