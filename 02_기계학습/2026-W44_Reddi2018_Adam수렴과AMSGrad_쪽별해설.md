# p.1 해설 — 초록과 서론: Adam이 수렴하지 않는 경우

**① 초록** — RMSProp, Adam, AdaDelta, NAdam은 모두 **과거 기울기 제곱의 지수이동평균의 제곱근**으로 기울기를 나눈다. 출력 공간이 큰 학습 등에서 이들이 최적해(비볼록이면 임계점)로 **수렴하지 못하는** 현상이 관찰되었다.
→ 🔎 *exponential moving averages of squared past gradients* 에 밑줄.

**② 원인 지목 ⭐** — 실패의 한 원인은 바로 **지수이동평균**. 단순한 **볼록** 문제에서 Adam이 최적해로 가지 않는 명시적 예 + 기존 Adam 분석의 정확한 문제점 제시.

**③ 해결 방향** — 과거 기울기에 대한 **"장기 기억"** 을 주면 고쳐진다 → 새 Adam 변형(AMSGrad).
→ 🔎 *"long-term memory"* 에 밑줄.

**④ 서론: 계보** — SGD → 좌표별로 학습률을 자동 조절하는 **AdaGrad**(Duchi 외 2011): 기울기가 **희소**할 때 SGD보다 훨씬 좋다.

**⑤ AdaGrad의 약점** — 비볼록·조밀한 기울기에선 **모든** 과거 기울기를 쓰므로 학습률이 너무 **빨리 감소**. 고차원 딥러닝에서 특히 심각.

**⑥ 지수이동평균의 등장과 부작용** — RMSProp, Adam 등은 **최근 몇 개** 기울기에만 의존해 감소를 완화. 그런데 어떤 미니배치는 **드물게 큰** 기울기를 주는데, 그 정보가 지수평균 때문에 **빨리 사라져** 수렴이 나빠진다.
→ 🔎 *their influence dies out rather quickly* 에 밑줄.

→ ✍️ 지수이동평균 $v_t=0.9v_{t-1}+0.1g_t^2$ 에서 $t=1$에 $g=10$, 이후 계속 $g=1$이면 $v_t$가 어떻게 변하는지 5스텝 계산해 보자.

<div style="page-break-after: always"></div>

# p.2 해설 — 기여와 온라인 최적화 설정

**① 기여 1** — RMSProp·Adam이 **증명 가능하게** 최적해로 가지 않는 단순 볼록 예. AdaDelta, NAdam, 그리고 사실상 **고정 크기 창**으로 기울기 제곱을 평균하는 모든 알고리즘에 확장된다.

**② 기여 2 ⭐** — 수렴을 보장하려면 **장기 기억**이 필요. Kingma & Ba(2015)의 **수렴 증명의 문제**를 지적. Adam과 **같은 시간·메모리**로 구현되는 새 변형 + AdaGrad 같은 **데이터 의존 regret** 경계.
→ 🔎 *we point out a problem with the proof of convergence* 에 밑줄.

**③ 기여 3** — 예비 실험: 비슷하거나 더 좋음.

**④ 표기법** — $\mathcal S^d_+$: 양정치 $d\times d$ 행렬, $\sqrt{\mathbf a}$, $\mathbf a^2$, $\mathbf a/\mathbf b$, $\max(\mathbf a,\mathbf b)$는 모두 **원소별**. 사영 $\Pi_{\mathcal F,A}(\mathbf y)=\arg\min_{\mathbf x\in\mathcal F}\|A^{1/2}(\mathbf x-\mathbf y)\|$. 지름 $D_\infty$.

**⑤ 온라인 최적화 ⭐** — 매 시각 $t$: 점 $\mathbf x_t$ 선택 → 손실 $f_t$ 공개(= 다음 미니배치의 손실) → $f_t(\mathbf x_t)$ 만큼 손해.
$$R_T=\sum_{t=1}^T f_t(\mathbf x_t)-\min_{\mathbf x\in\mathcal F}\sum_{t=1}^T f_t(\mathbf x)$$
목표 $R_T=o(T)$ = 평균적으로 최적 성능에 수렴.

**⑥ 기본 알고리즘** — 온라인 경사하강(Zinkevich 2003): $\mathbf x_{t+1}=\Pi_{\mathcal F}(\mathbf x_t-\alpha_t\mathbf g_t)$, $\alpha_t=\alpha/\sqrt t$. 평균 regret이 사라지는 온라인 알고리즘 ⇒ **경험적 위험 최소화(ERM)** 의 확률적 알고리즘 → "온라인 GD = SGD"로 취급.

→ ✍️ regret을 "시험을 볼 때마다 점수를 매기는데, 처음부터 정답 하나를 알고 있던 사람과의 누적 점수 차"로 비유해 설명해 보자.

<div style="page-break-after: always"></div>

# p.3 해설 — 적응형 방법의 공통 틀 (Algorithm 1)

**① Algorithm 1 ⭐**
$$\mathbf g_t=\nabla f_t(\mathbf x_t),\ \ \mathbf m_t=\phi_t(\mathbf g_1..\mathbf g_t),\ \ V_t=\psi_t(\mathbf g_1..\mathbf g_t),\ \ \mathbf x_{t+1}=\Pi_{\mathcal F,\sqrt{V_t}}(\mathbf x_t-\alpha_t\mathbf m_t/\sqrt{V_t})$$
$\alpha_t$ = **스텝 크기**, $\alpha_tV_t^{-1/2}$ = **학습률**. 대각 $V_t=\mathrm{diag}(\mathbf v_t)$만 고려.

**② SGD** — $\phi_t=\mathbf g_t$, $\psi_t=I$, $\alpha_t=\alpha/\sqrt t$. 감소 스텝은 수렴엔 필요하지만 실전 성능은 나쁨.

**③ AdaGrad** — $\psi_t=\mathrm{diag}(\sum_{i=1}^t\mathbf g_i^2)/t$, $\alpha_t=\alpha/\sqrt t$ → 실효 학습률 $\alpha/\sqrt{\sum_i g_{i,j}^2}$ (완만한 감소). 희소 기울기에서 큰 이득.

**④ Adam ⭐**
$$m_{t,i}=\beta_1m_{t-1,i}+(1-\beta_1)g_{t,i},\qquad v_{t,i}=\beta_2v_{t-1,i}+(1-\beta_2)g_{t,i}^2\quad (1)$$
추천값 $\beta_1=0.9$, $\beta_2=0.999$. $\mathcal F=\mathbb R^d$이면 사영은 항등 → 원래 Adam. **RMSProp** = $\beta_1=0$인 Adam. 딥러닝에선 $\beta_1\neq0$의 **모멘텀**이 성능을 크게 올림.
→ 🔎 식 (1)에 밑줄. 지난주 Kingma & Ba 해설의 식과 비교!

**⑤ 각주 1** — 단순화를 위해 **편향 보정(debiasing)** 단계를 뺐지만, 논증은 보정판에도 적용된다.

→ ✍️ SGD / AdaGrad / Adam을 $(\phi_t, \psi_t)$ 표로 정리하고, "과거를 얼마나 기억하는가"로 순위를 매겨 보자.

<div style="page-break-after: always"></div>

# p.4 해설 — Adam의 비수렴: Γ와 반례 ⭐⭐

**① 핵심량**
$$\Gamma_{t+1}=\frac{\sqrt{V_{t+1}}}{\alpha_{t+1}}-\frac{\sqrt{V_t}}{\alpha_t}\quad (2)$$
학습률 **역수**의 시간 변화. SGD·AdaGrad는 $\Gamma_t\succeq0$ (학습률 **비증가**). Adam·RMSProp은 $\Gamma_t$가 **부정부호**가 될 수 있다.
→ 🔎 식 (2)에 크게 밑줄.

**② 반례** — $\mathcal F=[-1,1]$, $C>2$:
$$f_t(x)=\begin{cases}Cx & t\bmod3=1\\ -x & \text{그 외}\end{cases}$$
최소 regret은 $x=-1$. $\beta_1=0$, $\beta_2=1/(1+C^2)$인 Adam은 **$x=+1$** 로 수렴!

**③ 직관** — 3스텝에 한 번 큰 기울기 $C$, 나머지 두 번은 $-1$(잘못된 방향). 큰 기울기가 $\sqrt{v}$로 나뉘어 거의 **$C$배 축소** → 잘못된 방향 두 번을 이기지 못함.

**④ 정리 1** — Adam의 평균 regret이 0으로 가지 않는 온라인 볼록 문제가 존재.

**⑤ ε을 더하면?** — $\hat{\mathbf x}_{t+1}=\mathbf x_t-\alpha_t\mathbf m_t/\sqrt{V_t+\epsilon I}$ (3). 어떤 상수 $\epsilon>0$에 대해서도 반례 존재 (정리 6, 부록 F).

**⑥ 파국적** — 구간에서 **가장 나쁜 점**으로 수렴. 상수 스텝에서도 성립. SGD·AdaGrad는 이 문제 없음. 고차원, 기울기 **분산**이 클 때 악화. 큰 $\beta_2$가 권장되는 이유이기도 함.

**⑦ 정리 2** — $\beta_1<\sqrt{\beta_2}$인 **모든** 상수 $\beta_1,\beta_2$에 대해 반례 존재. (이 조건은 실전에서도, Kingma & Ba 증명에서도 가정됨)

→ ✍️ $C=3$, $\beta_2=0.1$로 Adam($\beta_1=0$)을 $x_1=1$에서 6스텝 손으로 따라가 보자 (사영 포함).

<div style="page-break-after: always"></div>

# p.5 해설 — AMSGrad 알고리즘

**① Algorithm 2 AMSGrad ⭐⭐**
$$m_t=\beta_{1t}m_{t-1}+(1-\beta_{1t})g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$
$$\hat v_t=\max(\hat v_{t-1},v_t),\qquad x_{t+1}=\Pi_{\mathcal F,\sqrt{\hat V_t}}\big(x_t-\alpha_tm_t/\sqrt{\hat v_t}\big)$$
→ 🔎 `max` 한 줄에 크게 밑줄 — Adam과의 **유일한** 차이.

**② 정리 3** — **확률적** 볼록 문제에서도 Adam이 최적해로 가지 않는 예가 있다.

**③ 결과의 의미** — 피하려면 **문제마다** $\epsilon,\beta_1,\beta_2$를, 고차원이면 **차원마다** 다르게 튜닝해야 → 적응형 방법의 취지를 무너뜨림. 반례는 인위적이지만, 실제로 수렴을 **느리게** 할 수 있는 상황은 충분히 상상 가능.

**④ $\beta_1$ 감소에 대해** — Kingma & Ba는 $\beta_1$을 시간에 따라 줄이지만, 핵심 변수는 $\beta_2$이므로 $\beta_2$가 1에서 떨어져 있으면 분석 그대로 성립.

**⑤ 4절 설계 원리** — Adam의 원 증명은 $\Gamma_t$가 양반정치라고 **잘못 가정**(부록 D). 해결책 두 가지: (a) 알고리즘을 바꿔 제약 충족(AMSGrad), (b) $\beta_1,\beta_2$를 시간에 따라 바꾸기(AdamNC).

**⑥ 세 알고리즘 비교** — $v_{t-1,i}>g_{t,i}^2>0$일 때: Adam은 학습률을 **공격적으로 키움**(해로움), AdaGrad는 **살짝 줄임**(장기 누적으로 너무 작아짐), AMSGrad는 **키우지도 줄이지도 않음**.
→ ✍️ 이 세 반응을 표로 정리해 보자.

<div style="page-break-after: always"></div>

# p.6 해설 — 정리 4: AMSGrad의 regret 경계

**① 그림 1** — 1차원 합성 문제: 왼쪽·가운데는 온라인, 오른쪽은 확률적 설정에서 Adam vs AMSGrad.

**② 정리 4 ⭐** — $\alpha_t=\alpha/\sqrt t$, $\beta_{1t}\le\beta_1$, $\gamma=\beta_1/\sqrt{\beta_2}<1$, 지름 $D_\infty$, $\|\nabla f_t\|_\infty\le G_\infty$이면
$$R_T\le\frac{D_\infty^2\sqrt T}{\alpha(1-\beta_1)}\sum_{i=1}^d\hat v_{T,i}^{1/2}+\frac{D_\infty^2}{(1-\beta_1)^2}\sum_{t=1}^T\sum_{i=1}^d\frac{\beta_{1t}\hat v_{t,i}^{1/2}}{\alpha_t}+\frac{\alpha\sqrt{1+\log T}}{(1-\beta_1)^2(1-\gamma)\sqrt{1-\beta_2}}\sum_{i=1}^d\|g_{1:T,i}\|_2$$
→ 세 항의 의미: (1) 초기 거리 × $\sqrt T$, (2) 모멘텀 비용, (3) 기울기 크기.

**③ 따름정리 1** — $\beta_{1t}=\beta_1\lambda^{t-1}$이면 둘째 항이 상수 $\frac{\beta_1D_\infty^2G_\infty}{(1-\beta_1)^2(1-\lambda)^2}$.

**④ 해석** — $\sum_i\hat v_{T,i}^{1/2}\ll\sqrt d$, $\sum_i\|g_{1:T,i}\|_2\ll\sqrt{dT}$ 이면 SGD의 $O(\sqrt{dT})$보다 **훨씬 좋다**(희소 기울기). $\beta_{1t}=\beta_1/t$로도 $O(\sqrt T)$.

**⑤ 대안** — max 대신 과거 $v_t$의 **단순 평균**을 쓰면 AdaGrad와 비슷한 알고리즘.

**⑥ 5절 실험 시작** — 볼록(로지스틱 회귀)·비볼록(신경망) 다중 분류. 합성 문제:
$$f_t(x)=\begin{cases}1010x & t\bmod101=1\\-10x&\text{그 외}\end{cases},\quad \mathcal F=[-1,1],\ \beta_1=0.9,\ \beta_2=0.99$$
→ ✍️ 101스텝 동안의 기울기 합은? ($1010-100\cdot10=10>0$ → 최적해는 $x=-1$)

<div style="page-break-after: always"></div>

# p.7 해설 — 실험: 합성·MNIST·CIFAR-10

**① 그림 2** — 위: MNIST 로지스틱 회귀(학습/시험 손실), 1층 신경망. 아래: CIFARNET 학습·시험 손실.

**② 합성 결과 ⭐** — Adam: 평균 regret이 0으로 안 감, $x_t\to1$ (regret **최대** 점). AMSGrad: 평균 regret → 0, $x_t\to-1$ (최적).
확률적 버전: 확률 0.01로 $1010x$, 아니면 $-10x$ → 역시 Adam은 $x=1$로.
→ 🔎 *converge to x = 1, which unfortunately has the largest regret* 에 밑줄.

**③ 로지스틱 회귀 (볼록)** — MNIST 784차원 → 10클래스. $\alpha_t=\alpha/\sqrt t$ (이론과 일치), 미니배치 128, $\beta_1=0.9$, $\beta_2\in\{0.99,0.999\}$, 격자 탐색. AMSGrad가 학습·시험 손실 모두 우수, 하이퍼파라미터에 **더 견고**.

**④ 신경망 (비볼록)** — 은닉층 1개(ReLU 100개), **상수** $\alpha_t=\alpha$ (딥러닝 관행).

**⑤ CIFAR-10 / CIFARNET** — 32×32 이미지 6만 장. 합성곱 2층(64채널, 6×6 커널) + 완전연결 384, 192. 2×2 맥스풀링, 국소 응답 정규화, 드롭아웃 0.5, 미니배치 128.
→ ai_life **03 CNN** 단원의 구조 그림과 비교해 보자.

→ ✍️ ai_life **01 선형회귀** 데모에 Adam/AMSGrad 토글을 넣는다면, 학생에게 어떤 데이터를 보여 줘야 차이가 드러날까?

<div style="page-break-after: always"></div>

# p.8 해설 — AdamNC와 결론

**① CIFAR 결과** — AMSGrad가 학습 손실·정확도에서 상당히 우수, 시험 손실에도 이어짐.

**② 5.1 AdamNC** — 구조는 Adam 그대로, $\beta_2$를 **증가** 스케줄로 (Algorithm 3, 부록).

**③ 정리 5** — 조건 (1) $v$가 기울기 제곱합의 일정 비율 이상, (2) $v_{t,i}^{1/2}/\alpha_t\ge v_{t-1,i}^{1/2}/\alpha_{t-1}$ (= $\Gamma_t\succeq0$)이면 AMSGrad와 비슷한 regret 경계.

**④ 따름정리 2 ⭐** — $\beta_{2t}=1-1/t$이면 $v_{t,i}=\sum_{j=1}^tg_{j,i}^2/t$ → **모멘텀 붙은 AdaGrad**! 데이터 의존 regret.
→ 🔎 $\beta_{2t}=1-1/t$ 에 밑줄. "지수평균"을 "단순평균"으로 바꾸는 마법.

**⑤ 6절 결론 ⭐** — 본질적으로 **고정 크기 창**의 과거 기울기로 스케일하는 모든 알고리즘은 이 문제를 겪는다. 장기 기억을 주는 작은 수정으로 해결, 실전 성능도 유지·개선.
→ 🔎 *any algorithm that relies on an essentially fixed sized window* 에 밑줄.

**⑥ 메시지** — RMSProp·Adam은 크게 성공했으니 그 행동을 **엄밀히 이해**하고 **함정**을 알고 써야 한다.

→ ✍️ "고정 크기 창"의 비유: 최근 일주일 날씨만 보고 우산을 챙기는 사람과 1년 기록을 보는 사람 — 장마철 직후에는 누가 더 틀릴까?

<div style="page-break-after: always"></div>

# p.9 해설 — 참고문헌

**참고문헌 중 주목할 것**

| 참고문헌 | 왜 볼까 |
|---|---|
| Duchi, Hazan, Singer (2011) | **AdaGrad** — 적응형 방법의 출발점 |
| Kingma & Ba (2015) | **Adam** — 지난주 읽은 논문, 이번 반례의 대상 |
| Tieleman & Hinton (2012) | **RMSProp** — 코세라 강의 슬라이드가 출처 |
| Zeiler (2012) | AdaDelta |
| Dozat (2016) | NAdam (네스테로프 + Adam) |
| Zinkevich (2003) | 온라인 볼록 최적화와 온라인 경사하강 |
| McMahan & Streeter (2010) | 적응적 경계 최적화 (보조정리 4의 출처) |
| Auer & Gentile (2000) | 보조정리 5의 출처 |
| Krizhevsky 외 (2012) | AlexNet — CIFARNET 정규화의 출처 |

**✍️ 본문 정리 질문 (부록 들어가기 전에)**
1. $\Gamma_t$의 정의를 쓰고, SGD에서 $\Gamma_t\succeq0$ 임을 보이자 ($\sqrt{V_t}=I$, $\alpha_t=\alpha/\sqrt t$).
2. Adam에서 $\Gamma_t$가 음수가 되는 구체적 상황을 한 줄로 (큰 $g$ 뒤에 작은 $g$들이 이어질 때 $v_t$가 줄어든다).
3. AMSGrad의 max가 $\Gamma_t\succeq0$을 보장하는 이유는? (상수 $\beta_2$, 감소하는 $\alpha_t$와 함께)

<div style="page-break-after: always"></div>

# p.10 해설 — 부록 A: 정리 1의 증명 (1)

**① 설정** — $f_t$ 선형, $\mathcal F=[-1,1]$, $C\ge2$, $x_1=1$ (좌표 평행이동으로 일반성 유지).
$$\beta_1=0,\quad \beta_2=\frac1{1+C^2},\quad \alpha_t=\frac{\alpha}{\sqrt t},\quad \alpha<\sqrt{1-\beta_2}$$
Kingma & Ba의 조건(유계 기울기, 유계 지름, $\beta_1^2/\sqrt{\beta_2}<1$) 모두 충족 — 즉 **원 논문의 가정 안에서** 반례.
→ 🔎 *the conditions on the parameters required for ADAM are satisfied* 에 밑줄.

**② 주장 ⭐** — 모든 $t$에서 $x_t>0$, 그리고 $x_{3t+1}=1$. **수학적 귀납법**으로.

**③ 기울기** — $\nabla f_i=C$ ($i\bmod3=1$), 아니면 $-1$.

**④ (3t+1)번째 갱신**
$$\hat x_{3t+2}=1-\frac{\alpha C}{\sqrt{(3t+1)(\beta_2v_{3t}+(1-\beta_2)C^2)}}$$
분모에서 $\beta_2v_{3t}\ge0$을 버리면
$$\frac{\alpha C}{\sqrt{(3t+1)(1-\beta_2)C^2}}=\frac{\alpha}{\sqrt{(3t+1)(1-\beta_2)}}<1\quad(4)$$
→ $0<x_{3t+2}<1$. **큰 기울기 $C$가 $\sqrt{C^2}$로 나뉘어 상쇄**되는 것이 핵심!
→ 🔎 식 (4)에 밑줄.

**⑤ 다음 두 갱신** — 기울기 $-1$이므로 $x$가 **증가**: $\hat x_{3t+3}=x_{3t+2}+\dfrac{\alpha}{\sqrt{(3t+2)(\beta_2v_{3t+1}+1-\beta_2)}}$, $\hat x_{3t+4}$도 마찬가지.

**⑥ 목표** — $\hat x_{3t+4}\ge1$을 보이면 사영으로 $x_{3t+4}=1$.

→ ✍️ 왜 $-1$ 기울기 때의 분모는 작고(따라서 이동량이 크고), $C$ 기울기 때의 분모는 큰가? $v$에 $C^2$이 들어가는 시점을 표시해 보자.

<div style="page-break-after: always"></div>

# p.11 해설 — 부록 A: 정리 1의 증명 (2)

**① 경우 나누기** — $\hat x_{3t+3}\ge1$이면 바로 $\hat x_{3t+4}>1$. 아니면 다음을 보이면 충분:
$$T_1=\frac{\alpha C}{\sqrt{(3t+1)(\beta_2v_{3t}+(1-\beta_2)C^2)}}\ \le\ T_2=\frac{\alpha}{\sqrt{(3t+2)(\beta_2v_{3t+1}+1-\beta_2)}}+\frac{\alpha}{\sqrt{(3t+3)(\beta_2v_{3t+2}+1-\beta_2)}}$$
= "한 번 뒤로 간 거리 ≤ 두 번 앞으로 간 거리".

**② $T_1$ 위로** — 식 (5): $T_1\le\dfrac{\alpha}{\sqrt{(3t+1)(1-\beta_2)}}$.

**③ $T_2$ 아래로** — $v_t\le C^2$ 사용, $\frac1{\sqrt{3t+2}}+\frac1{\sqrt{3t+3}}\ge\frac{\sqrt2}{\sqrt{3t+1}}$ 류의 부등식으로
$$T_2\ge\frac{\sqrt2\,\alpha}{\sqrt{(3t+1)(\beta_2C^2+1-\beta_2)}}=\frac{\alpha}{\sqrt{(3t+1)(1-\beta_2)}}\ge T_1\quad(6)$$
마지막 등호: $\beta_2=1/(1+C^2)$이면 $\dfrac{\beta_2C^2+(1-\beta_2)}{2}=1-\beta_2$.
→ ✍️ 이 등식을 직접 확인해 보자. ($\beta_2C^2+1-\beta_2=\frac{C^2+C^2}{1+C^2}=\frac{2C^2}{1+C^2}$, $1-\beta_2=\frac{C^2}{1+C^2}$)

**④ 결론 ⭐** — 귀납법으로 모든 $t$에서 $x_{3t+1}=1$. 3스텝마다 regret
$$\ge 2C-4$$
(첫 스텝: $f(1)-f(-1)=C-(-C)=2C$; 나머지 두 스텝: $-x-1\ge-2$ 씩, 합 $\ge-4$). 따라서 $R_T\ge(2C-4)T/3$, $R_T/T\not\to0$.
→ 🔎 *ADAM suffers a regret of at least 2C − 4* 에 밑줄.

→ ✍️ $C=2$이면 $2C-4=0$. 그래서 본문은 $C>2$를 요구한다. 왜 $C=2$는 반례가 안 되는지 생각해 보자.

<div style="page-break-after: always"></div>

# p.12 해설 — 부록 B: 정리 2의 증명 (1)

**① 일반화** — 이제 $\beta_1,\beta_2$가 **임의의 상수**($\beta_1<\sqrt{\beta_2}$). 함수열:
$$f_t(x)=\begin{cases}Cx & t\bmod C=1\\ -x &\text{그 외}\end{cases}$$
$C$는 짝수 자연수, 식 (7)의 세 조건을 만족할 만큼 **큼** ($\gamma=\beta_1/\sqrt{\beta_2}<1$).
→ 주기가 3에서 $C$로: 큰 기울기가 **더 드물게** 온다.

**② 관찰: $m_{kC}\le0$** — 주기 끝에서 모멘텀이 항상 0 이하.
$$m_{kC+C}=-(1-\beta_1^{C-1})+(1-\beta_1)\beta_1^{C-1}C+\beta_1^Cm_{kC}\quad(9)$$
$C$가 크면 $\beta_1^{C-1}C$가 작아 귀납적으로 $\le0$.

**③ 전략** — $T'$ 이후( $t+C\le\tau^2t$, $\tau\le3/2$ )만 분석. 그 전 regret은 상수라 평균엔 무관.

**④ 주장 (10)** — $t=kC$에서 $x_{t+C}\ge\min\{x_t+c_t,1\}$, $c_t>0$. 즉 한 주기가 지나면 $x$가 **오른쪽(잘못된 쪽)으로** 순증가.

**⑤ 이동량 $\delta$** — 첫 스텝 $\delta_t$(기울기 $C$, 음수 이동), 나머지 $\delta_{t+i}$(기울기 $-1$). 한 번 양수가 되면 계속 양수 → **보조정리 6**(1차원 사영 누적)으로 $x_{t+C}\ge\min\{1,x_t+\sum\delta_i\}$.

→ ✍️ 정리 1(주기 3)과 정리 2(주기 $C$)의 차이를 한 문장으로: 무엇을 크게 해서 임의의 $\beta$를 이기는가?

<div style="page-break-after: always"></div>

# p.13 해설 — 부록 B: 정리 2의 증명 (2)

**① 목표** — $\delta=\sum_{i=t}^{t+C-1}\delta_i>0$.

**② 기울기 $-1$ 구간의 합을 아래로** — 정의에 $m$을 풀어 넣고, $m_t\le0$, $t+C\le\tau^2t$, $v_{t+i-1}\ge(1-\beta_2)\beta_2^{i-2}C^2$ 등을 차례로 사용.

**③ 핵심 상계 ⭐** — 주기 후반($i'=C/2\le i\le C$)에선 큰 기울기 $C^2$의 흔적이 $\beta_2^{i'-1}$배로 **거의 사라져**
$$v_{t+i-1}\le(1-\beta_2)\Big(\beta_2^{i'-1}C^2\frac{1}{1-\beta_2^C}+\frac1{1-\beta_2}\Big)\le2$$
→ 🔎 이 부등식의 의미에 밑줄: "지수평균은 큰 기울기를 **잊는다**" — 서론의 직관이 수식이 된 곳.

**④ 결과** — 후반 $C-i'$ 스텝 각각이 $\dfrac{\alpha}{\tau\sqrt t\sqrt{1+\beta_2}}$ 이상 오른쪽으로 → 합이 첫 스텝의 왼쪽 이동을 이긴다.

**⑤ 사용된 부등식들** — 기하급수 합 $\sum\beta^j\le\frac1{1-\beta}$, $\gamma$에 대한 합 $\frac{\gamma(1-\gamma^{C-1})}{1-\gamma}$.

→ ✍️ $\beta_2=0.999$, $C=10000$이면 $\beta_2^{C/2-1}\approx e^{-5}\approx0.007$. 즉 5000스텝 전의 큰 기울기는 $v$에 0.7%만 남는다. 실전 $\beta_2$에서도 문제가 생길 수 있는 이유!

<div style="page-break-after: always"></div>

# p.14 해설 — 정리 2 마무리, 부록 C: 확률적 반례

**① λ** — $\sum\delta_i\ge\lambda/\sqrt t$, $\lambda\ge0$는 $t$와 무관. 따라서 $x_{t+C}\ge\min\{1,x_t+\lambda/\sqrt t\}$.

**② 두 사실** — (1) $x_t=1$이면 이후 주기 끝마다 1. (2) $\sum1/\sqrt t$는 **발산**하므로 언젠가 $x=1$ 도달.

**③ regret** — 그 이후 $C$스텝마다 regret $\ge2C-2(C-1)=2$ → $R_T\ge2(T-T_1')/C$, 평균 regret이 0으로 안 감.

**④ 부록 C: 정리 3 ⭐** — 매 시각 i.i.d.로
$$f_t(x)=\begin{cases}Cx & \text{확률 } p=\frac{1+\delta}{C+1}\\ -x & \text{확률 } 1-p\end{cases}$$
기댓값 함수 $F(x)=\delta x$ → 최적점 $x^\star=-1$.
→ ✍️ $F(x)=pCx-(1-p)x$ 를 계산해 $\delta x$가 나오는지 확인하자.

**⑤ Adam의 스텝**
$$\Delta_t=-\alpha_t\frac{\beta_1m_{t-1}+(1-\beta_1)g_t}{\sqrt{\beta_2v_{t-1}+(1-\beta_2)g_t^2}}$$

**⑥ 보조정리 1** — $C$가 충분히 크면 $\mathbb E[\Delta_t]\ge0$ → Adam의 스텝이 **평균적으로 최적해에서 멀어진다**.
→ 🔎 *keep drifting away from the optimal solution* 에 밑줄.

<div style="page-break-after: always"></div>

# p.15 해설 — 보조정리 1의 증명

**① 기댓값 분해 (11)** — $\mathbb E_t[\Delta_t]/\alpha_t = T_1 + T_2 + T_3$:
- $T_1$: 큰 기울기 $C$가 왔을 때의 (음의) 이동
- $T_2$: 모멘텀 $-\beta_1m_{t-1}$ 의 기여
- $T_3$: 기울기 $-1$일 때의 (양의) 이동

**② $T_1$ 아래로 (12)** — $T_1\ge-\dfrac{1}{\sqrt{1-\beta_2}}$ (분모에 $C^2$이 있어 $C$가 상쇄). 이 항에 확률 $p\approx1/C$가 곱해지므로 작다.

**③ $T_2$ (13)** — $k=\lceil\log(C+1)/\log(1/\beta_1)\rceil$. 최근 $k$번 모두 $-1$인 사건 $E$ (확률 $\ge1-kp$)에서는 $m_{t-1}\le0$ → $T_2\ge0$. 아니면 **코시–슈바르츠**로 $m_{t-1}\le A\sqrt{v_{t-1}}$ 를 써서 $\mathbb E[T_2]\ge-\dfrac{\beta_1(1-\beta_1)kp}{\sqrt{(1-\beta_2)(\beta_2-\beta_1^2)}}$.

**④ $T_3$ — 옌센 부등식** — $1/\sqrt x$는 볼록 → $\mathbb E[T_3]\ge\dfrac{1-\beta_1}{\sqrt{\beta_2(1+\delta)C+(1-\beta_2)}}$.

**⑤ (14)** — $\mathbb E[v_{t-1}]=(1-\beta_2^{t-1})(C^2p+1-p)\le(1+\delta)C$.

→ ✍️ 크기 비교: $T_1$ 기여 $\sim p\cdot1=O(1/C)$, $T_2$ 기여 $\sim kp=O(\log C/C)$, $T_3$ 기여 $\sim1/\sqrt C$. $C\to\infty$에서 누가 이기나? ($1/\sqrt C$가 가장 천천히 줄어든다 → 양수!)

<div style="page-break-after: always"></div>

# p.16 해설 — 정리 3 마무리, 부록 D: AMSGrad 수렴 증명 (1)

**① 정리 3 결론** — 세 경계를 합치면 $C$가 크면 $\mathbb E[\Delta_t]\ge0$. 사영 없는 Adam이면 $\mathbb E[x_{t+1}]\ge\mathbb E[x_t]$ → $x_1\ge0$에서 시작하면 $\mathbb E[F(x_t)]\ge0$, 그런데 $F(-1)=-\delta$. **기대 최적성 격차가 항상 $\delta$**.

**② 부록 D의 위치 ⭐** — Kingma & Ba 정리 4.1을 따라가되, 그 증명의 문제: (a) $\Gamma_t$가 양반정치라는 **잘못된 가정**, (b) 그들의 **보조정리 10.3, 10.4**의 문제. 이 증명이 그것을 고친다.
→ 🔎 *incorrect assumption that Γt … is positive semidefinite* 에 밑줄.

**③ 출발점** — 사영의 **비확장성**(보조정리 4)으로
$$\|\hat V_t^{1/4}(x_{t+1}-x^*)\|^2\le\|\hat V_t^{1/4}(x_t-x^*)\|^2+\alpha_t^2\|\hat V_t^{-1/4}m_t\|^2-2\alpha_t\langle m_t,x_t-x^*\rangle$$

**④ 정리 (15)** — $\langle g_t,x_t-x^*\rangle$ 를 "거리 감소량 + 스텝 크기 항 + 모멘텀 항"으로 상계. 코시–슈바르츠와 **영(Young) 부등식** 사용.

→ ✍️ 영 부등식 $ab\le\frac{a^2}{2\epsilon}+\frac{\epsilon b^2}{2}$ 를 써서 $\langle m_{t-1},x_t-x^*\rangle$ 같은 내적을 쪼개는 연습을 해 보자.

<div style="page-break-after: always"></div>

# p.17 해설 — 부록 D (2): 볼록성과 보조정리 2

**① 볼록성으로 regret 쪼개기 (16)**
$$\sum_{t=1}^T f_t(x_t)-f_t(x^*)\le\sum_{t=1}^T\langle g_t,x_t-x^*\rangle$$
(볼록함수는 접선 위에 있다: $f(x^*)\ge f(x_t)+\langle g_t,x^*-x_t\rangle$) → 이후 (15)를 대입. $\hat v_{t-1,i}\le\hat v_{t,i}$ (**max 덕분**)를 사용.
→ 🔎 *the fact that $\hat v_{t-1,i}\le\hat v_{t,i}$* 에 밑줄 — Adam에선 이것이 성립하지 않는다!

**② 보조정리 2 ⭐**
$$\sum_{t=1}^T\alpha_t\|\hat V_t^{-1/4}m_t\|^2\le\frac{\alpha\sqrt{1+\log T}}{(1-\beta_1)(1-\gamma)\sqrt{1-\beta_2}}\sum_{i=1}^d\|g_{1:T,i}\|_2$$

**③ 증명 시작** — 마지막 항을 떼어 내고, $\hat v_{T,i}\ge v_{T,i}$ 로 분모를 바꾼 뒤 $m_T,v_T$를 기울기 합으로 전개.

**④ 코시–슈바르츠 (17)** — 분자 $(\sum\beta_1^{T-j}g_j)^2\le(\sum\beta_1^{T-j})(\sum\beta_1^{T-j}g_j^2)$, $\sum\beta_1^{T-j}\le\frac1{1-\beta_1}$ → 비 $\beta_1^{T-j}/\sqrt{\beta_2^{T-j}}=\gamma^{T-j}$ 등장.

→ ✍️ 왜 조건 $\gamma=\beta_1/\sqrt{\beta_2}<1$ 이 필요한지 이 단계에서 확인하자 (기하급수 $\sum\gamma^j$ 수렴).

<div style="page-break-after: always"></div>

# p.18 해설 — 부록 D (3): 망원 합

**① 보조정리 2 마무리** — 합의 순서를 바꾸고 $\sum_{j\ge t}\gamma^{j-t}\le\frac1{1-\gamma}$, 코시–슈바르츠, **조화급수** $\sum_{t=1}^T1/t\le1+\log T$.
→ 🔎 *harmonic sum* 에 밑줄 — $\sqrt{1+\log T}$의 출처.

**② 정리 4로 복귀 (18)** — 보조정리 2를 (16)에 넣고, 첫 항을
$$\frac{\|\hat V_t^{1/4}(x_t-x^*)\|^2}{\alpha_t}-\frac{\|\hat V_{t-1}^{1/4}(x_t-x^*)\|^2}{\alpha_{t-1}}$$
꼴로 재배열 → 좌표별로 $(x_{t,i}-x_i^*)^2\Big(\frac{\hat v_{t,i}^{1/2}}{\alpha_t}-\frac{\hat v_{t-1,i}^{1/2}}{\alpha_{t-1}}\Big)$.

**③ 바로 여기가 Γ ⭐⭐** — 괄호 안이 $\Gamma_t$의 좌표 성분. AMSGrad에서는 **항상 $\ge0$** 이라 $(x_{t,i}-x_i^*)^2\le D_\infty^2$로 바꿔 **망원 합**(telescoping)이 가능. Adam에선 음수일 수 있어 이 단계가 **무너진다**.
→ ✍️ 괄호가 음수일 때 $(x-x^*)^2\le D^2$로 바꾸면 왜 부등호 방향이 틀리는지 설명해 보자.

**④ 각주 3** — 원 증명에 빠졌던 상수 인자 $\frac{2}{1-\beta_1}$을 이번에 바로잡았다.

<div style="page-break-after: always"></div>

# p.19 해설 — 정리 4 완성, 부록 E: AdamNC

**① Algorithm 3 AdamNC** — Adam과 같은 구조, 단 $(\beta_{1t},\beta_{2t})$가 시간에 따라 변함:
$$m_t=\beta_{1t}m_{t-1}+(1-\beta_{1t})g_t,\quad v_t=\beta_{2t}v_{t-1}+(1-\beta_{2t})g_t^2$$

**② 망원 합 결과** — 
$$\sum_t\Big(\frac{\hat v_{t,i}^{1/2}}{\alpha_t}-\frac{\hat v_{t-1,i}^{1/2}}{\alpha_{t-1}}\Big)D_\infty^2=\frac{\hat v_{T,i}^{1/2}}{\alpha_T}D_\infty^2-\cdots$$
→ 정리 4의 첫 항 $\frac{D_\infty^2\sqrt T}{\alpha(1-\beta_1)}\sum_i\hat v_{T,i}^{1/2}$ 이 나온다.

**③ 최악의 경우** — $\sum_t|g_{t,i}|/\sqrt t=O(G_\infty\sqrt T)$ 이므로 AMSGrad의 regret은 $\min\{O(G_\infty\sqrt T),\ \text{정리 4 경계}\}$ → $T$에 대한 최악 의존도는 $O(\sqrt T)$.
→ 🔎 *the worst case dependence of regret on T in our case is O(√T)* 에 밑줄.

**④ 부록 E: 정리 5 증명 시작** — 정리 4와 같은 방법으로 (19)까지. 차이는 $\hat V_t$ 대신 $V_t$.

→ ✍️ $O(\sqrt T)$ regret이면 평균 regret은 $O(1/\sqrt T)$. $T=10^6$이면 평균 손해는 대략 몇 분의 1로 줄어드나?

<div style="page-break-after: always"></div>

# p.20 해설 — 부록 E (2): 보조정리 3

**① (20)** — 볼록성 + 정리 5의 조건 2 ($v_{t,i}^{1/2}/\alpha_t\ge v_{t-1,i}^{1/2}/\alpha_{t-1}$, 즉 **$\Gamma_t\succeq0$을 가정으로 요구**).

**② 보조정리 3** —
$$\sum_{t=1}^T\alpha_t\|V_t^{-1/4}m_t\|^2\le\frac{2\zeta}{(1-\beta_1)^2}\sum_{i=1}^d\|g_{1:T,i}\|_2$$

**③ 증명 (21)** — 코시–슈바르츠 → $\beta_{1k}\le\beta_1$ → $\sum\beta_1^{T-j}\le\frac1{1-\beta_1}$ → **조건 1**(분모가 기울기 제곱합의 $1/\zeta$배 이상)을 써서
$$\le\frac{\zeta}{1-\beta_1}\sum_{i}\sum_j\frac{\beta_1^{T-j}g_{j,i}^2}{\sqrt{\sum_{k\le j}g_{k,i}^2}}$$
→ 분모가 **누적합**(AdaGrad형)이 되는 것이 핵심.

→ ✍️ AMSGrad(정리 4)는 $\Gamma\succeq0$을 **알고리즘이 보장**하고, AdamNC(정리 5)는 **가정으로 요구**한다. 실전에서 어느 쪽이 편할까?

<div style="page-break-after: always"></div>

# p.21 해설 — 정리 5 마무리, 부록 F: ε으로도 안 된다

**① 보조정리 3 마무리** — 보조정리 5(Auer & Gentile): $\sum_i\frac{y_i}{\sqrt{\sum_{j\le i}y_j}}\le2\sqrt{\sum_iy_i}$ → $\frac{2\zeta}{(1-\beta_1)^2}\sum_i\|g_{1:T,i}\|_2$.

**② 정리 5 완성 (22)** — 정리 4와 같은 망원 합으로
$$R_T\le\frac{D_\infty^2}{2\alpha_T(1-\beta_1)}\sum_iv_{T,i}^{1/2}+\frac{D_\infty^2}{(1-\beta_1)^2}\sum_{t,i}\frac{\beta_{1t}v_{t,i}^{1/2}}{\alpha_t}+\frac{2\zeta}{(1-\beta_1)^3}\sum_i\|g_{1:T,i}\|_2$$

**③ 부록 F: 정리 6 ⭐** — 분모에 $\epsilon$을 더한 수정판 Adam(식 3)도, Kingma & Ba의 모든 조건을 만족하면서 평균 regret이 0으로 가지 않는 볼록 문제가 있다.
→ 🔎 *For any ϵ > 0* 에 밑줄 — 실무에서 "ε을 키우면 되지 않나?"에 대한 답.

→ ✍️ 보조정리 5를 $y_i=1$ (모두 1)로 확인해 보자: 좌변 $\sum1/\sqrt i\approx2\sqrt t$, 우변 $2\sqrt t$.

<div style="page-break-after: always"></div>

# p.22 해설 — 부록 F: 정리 6의 증명

**① 설정** — 먼저 $\epsilon=1$ (실은 $\epsilon\le1$ 모두). 정리 1과 **같은** 함수열($Cx$ / $-x$, 주기 3), $x_1=1$, 단
$$\beta_1=0,\quad\beta_2=\frac{2}{(1+C^2)C^2},\quad\alpha_t=\frac{\alpha}{\sqrt t},\quad\alpha<\sqrt{1-\beta_2}$$

**② 귀납법** — 정리 1과 동일한 구조: $x_{3t+2},x_{3t+3}>0$, $x_{3t+4}=1$.

**③ 차이점** — 모든 분모에 $+\epsilon$. 큰 기울기 스텝:
$$\hat x_{3t+2}=1-\frac{\alpha C}{\sqrt{(3t+1)(\beta_2v_{3t}+(1-\beta_2)C^2+\epsilon)}}\ge0$$

**④ 핵심 부등식 (23)** — 두 번의 앞 이동 ≥ 한 번의 뒤 이동. 이번엔 등식
$$\frac{\beta_2C^2+(1-\beta_2)+\epsilon}{2}\ \text{와}\ (1-\beta_2)+\frac{\epsilon}{C^2}$$
의 관계를 $\beta_2$ 선택으로 맞춘다 (원문 마지막 줄 참고).

**⑤ 결론** — 다시 3스텝마다 regret $\ge2C-4$.

→ ✍️ $\epsilon$이 분모에 들어가면 큰 기울기와 작은 기울기 중 누구의 스텝이 **상대적으로** 더 줄어드는가? 그래도 반례가 유지되는 이유를 직관적으로 설명해 보자.

<div style="page-break-after: always"></div>

# p.23 해설 — 일반 ε과 보조정리들, 전체 정리

**① 일반 $\epsilon$** — 함수열을 $\sqrt\epsilon$배로 **재조정**: $f_t=C\sqrt\epsilon x$ / $-\sqrt\epsilon x$ → $R_T\ge(2C-4)\sqrt\epsilon\,T/3$.

**② 보조정리 4** (McMahan & Streeter) — 양정치 $Q$에 대한 사영은 **비확장**: $\|Q^{1/2}(u_1-u_2)\|\le\|Q^{1/2}(z_1-z_2)\|$.
**보조정리 5** (Auer & Gentile) — $\sum_i\frac{y_i}{\sqrt{\sum_{j\le i}y_j}}\le2\sqrt{\sum_iy_i}$.
**보조정리 6** — 1차원 구간 사영에서 이동량이 "음수 구간 → 양수 구간" 순이면 $y_{T+1}\ge\min\{b,\ y_1+\sum\delta_j\}$.

**✍️ 논문 전체 정리 질문**
1. 정리 1의 반례를 "3스텝 이야기"로 친구에게 설명하자: 누가 몇 번, 어느 방향으로, 얼마나 움직이나?
2. Adam 원 증명의 오류는 정확히 어느 단계였나? (p.18의 망원 합, $\Gamma_t\succeq0$)
3. AMSGrad 코드는 Adam 코드에서 **몇 줄**을 바꾸면 되나? 직접 파이썬으로 써 보자.
4. 이상구 외(2023)의 경사하강법 → SGD → (지난주) Adam → AMSGrad 로 이어지는 "학습률 조절의 역사"를 한 장의 표로 정리하자.
5. "실험이 잘 되는 것"과 "수렴이 증명되는 것"은 어떻게 다른가? ai_life 학생들에게 이 논문을 어떤 이야기로 들려줄 수 있을까?
