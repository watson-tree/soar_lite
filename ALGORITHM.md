# 对角预条件 n_paths 路径搜索：算法与原理

本文描述**算法与不变式**，不限定接口。文中出现的类型名、函数名、参数名只是参考实现的一种拼法；
移植时尽可改名、改签名、改文件划分。第 6 节列出哪些东西是真的可换、哪些是不许破的。

设定：目标是一条批量可求值的黑箱函数——给一批点 `x`（形状 `(n, d)`），返回 `f`（`(n,)`）与
`∇f`（`(n, d)`），可选地返回一个 feasible 掩码。要最小化 feasible 点上的 `f`。没有 Hessian、没有
曲率估计，只有最近一对点差给出的 secant。预算按 eval 计（两种口径见第 5 节）。

## 1. 局部下降：逐坐标 secant 幅值比预条件

### 1.1 定义：割线方程的对角近似

准牛顿法从**割线方程** `B s = y` 出发（`B ≈ H`，`s = x_k − x_{k−1}`、`y = g_k − g_{k−1}`）。只留
它的对角、忽略耦合项（即假设 `yᵢ = Bᵢᵢ sᵢ`），直接得到

```text
Bᵢᵢ = yᵢ / sᵢ        ⇒        dᵢ := 1/Bᵢᵢ ≈ sᵢ / yᵢ
```

`dᵢ` 就是「这一坐标上，观察到的位移比上观察到的梯度变化」，量纲 `[x]²/[f]`，逐坐标独立。
（同一个对象也从 SR1 更新的对角退化掉出来：取 `B` 为对角时 SR1 同样退化为 `Bᵢᵢ = yᵢ/sᵢ`。结论一致，
但绕一道——割线方程是最短的那条路，本方法不需要 SR1 的完整更新式，也不需要存 `B`。）

单对 `(s, y)` 的 `sᵢ/yᵢ` 噪声大，所以沿轨迹用 EMA 平滑——**平滑作用在 secant 对的幅值上，不作用在
比值本身上**：

| 变体 | 分子 | 分母 | 逆曲率 | 步 |
| --- | --- | --- | --- | --- |
| `"abs"`（绝对值平均） | `S₁ = EMA(\|s\|)` | `Y₁ = EMA(\|y\|)` | `d = S₁ / max(Y₁, λ)` | `Δx = −d ⊙ g` |
| `"rms"`（均方根，默认） | `S₂ = EMA(s²)` | `Y₂ = EMA(y²)` | `d = √(S₂ / max(Y₂, λ))` | `Δx = −d ⊙ g` |

单对上两个变体是同一个值（`√(s²/y²) = |s/y|`），差别只在平滑哪个矩：`"abs"` 平均幅值、`"rms"` 平均幅值的平方
（RMS），前者对小 secant 更平滑、后者对大 secant 更敏感。EMA 窗口 ≈ `1/(1−β)` 步，默认 `β = 0.95`
⇒ 约 20 步。用幅值而非带符号的 `sᵢyᵢ` 是为了恒正：分母一旦能穿零或变负，`d ≥ 0` 就没有了。

**下降方向校正**：拿到 `d ≥ 0`，步取 `Δx = −diag(B)⁻¹ g = −d ⊙ g`。这一步的实质是把方向从
**梯度类量**改成**位移类量**——`−g` 的量纲是 `[f]/[x]`，直接当位移用，尺度会随坐标刻度反向走；
乘上 `d` 把 `1/H` 的量纲补进去。一次逐元素乘除完成，不需要矩阵、不需要解线性系统、不需要标量
学习率（第 1.4 节会用到「位移类」这个性质）。

### 1.2 三条不变式（这是本方法的全部价值）

1. **恒下降**：`d ≥ 0` 由构造保证（绝对值/平方/开方 + 正地板），所以 `(−d⊙g)·(−g) ≥ 0` 逐坐标成立。
   预条件后仍是下降方向 ⇒ **暖机之后的步不需要任何 acceptance test**，不需要线搜索去「验证方向」。
2. **单位不变**：`x̃ = c·x` ⇒ `d̃ = c²·d`、`Δx̃ = c·Δx`；`f̃ = c_f·f` ⇒ `d → d/c_f`、`Δx` 不变。
   这正好是「一个带物理量纲的步」该有的缩放，也让同一组无量纲旋钮能同时服务 ℃、m、Pa 这些量纲。
3. **无标量学习率**：全部旋钮无量纲（见第 6 节），没有 `[x]` 量纲的自由系数。相对 Adam 的关键差异：
   Adam 的 `m/√v` 无量纲、步长量纲全压在标量 `α` 上，多量纲问题里单一 `α` 无法同时是对的。

### 1.3 地板 `λ`：守卫，不是加数

`λ = θ·|gᵢ|`（`"abs"`，量纲 `[f]/[x]`）或 `λ = θ·gᵢ²`（`"rms"`，量纲 `[f]²/[x]²`），`θ` 无量纲。锚到**当前
梯度**而非 `y₀` 或 `|f_k|`：地板必须随 `g` 消失、而不是随 `y` 消失。平坦但未收敛的坐标（`y→0`
而 `g≠0`）靠它拿到有限步；真正收敛（`g→0`）由停止条件先接手，不会在 `g≈0` 处再除一次。

地板**用 `max` 而不是 `+`**：

```text
den = max(Y, λ)        ← 本实现
den = Y + λ            ← 被否掉的加法拼法
```

两者只在 `Y < λ` 时同义（都兑底 `Y→0`）。`Y ≥ λ` 时，加法拼法给出的 `d` 比 `max` 拼法小
`√(Y/(Y+λ))` 倍（`"rms"`）——也就是**在没有触底、曲率估计本来可靠的时候，λ 仍在持续压低步长**。
比值 `d_max/d_add = √((Y+λ)/max(Y,λ))` 在 `Y = λ` 处取峰值 `√2`（`"abs"` 因未开方，峰值是 `2`），
`Y→0` 与 `Y→∞` 两端都趋于 1。也就是说加法地板把 `λ` 混进了「步长调节」的职责，而你无法再由 `Y`
读出「这个坐标是不是曲率主导」。

`max` 把职责分开：**λ 只做 `Y→0` 的守卫**，`Y` 一旦高于 `λ`，`d` 就忠实等于 `S/Y`。步长该多激进
由独立旋钮承担（`alpha0` 全步阻尼、暖机 Armijo 的 `backtrack`、`n_alpha` 网格），不塞进地板里。这不是凭经验调出来的
收益，而是把估计量从「被 λ 污染的比值」还原成「比值本身」——解耦，而不是调参。

### 1.4 暖机步：唯一需要护栏的那一步

第一步没有 secant 对，`d` 无从估计，只能沿 `−g₀` 做一次**单调 Armijo 回溯**线搜索：初值 `α0`（默认 1，即全步阻尼旋钮），
每次回溯 `α ← α·backtrack`，充分下降判据用斜率 `armijo_c`。走出第一步，拿到首个 `(s, y)`，此后
`n_alpha = 1`（默认）无条件接受整步，超调交给下一条 secant 自校正。

#### 为什么只有暖机需要护栏，而稳态不需要

非对称的根源是有没有 `d`。

- 暖机的方向是 `−g₀`，**梯度类量**：`x̃ = c·x ⇒ g̃ = g/c`，坐标放大、梯度反而缩小。把它乘上标量
  `α` 直接塞进 `x + α·(−g)`，步长会随坐标刻度反向缩放，所以它本身是坐标依赖的；而回溯判据（`c`、
  `backtrack` 都无量纲）只保证「这一步比当前点好」，不保证尺度对。换任何 `α₀` 都改不了方向的缩放
  规律——只有换一个随坐标**同向**缩放的方向（位移类量）才行。为过渡一步去维护正定度量（牛顿方向
  `H⁻¹g` 正是那个对象）不值得，故接受「暖机步不追求坐标不变」。
- 稳态的方向是 `−d⊙g`，**位移类量**：`d` 已经把 `1/H` 的量纲补进去，`Δx` 随坐标同向缩放。它恒下
  降、坐标无关，所以不需要 acceptance test——护栏在这里只会是浪费的 eval。

一句话：**暖机没有 `d`，所以有护栏；稳态有 `d`，所以没有。** 这个非对称不是历史遗留。

#### 护栏与协变在第一步互斥

这不是「还没找到两全的办法」，而是结构上不能两全：护栏＝**所有坐标共用同一个标量** `α`（一次
下降判据判定全部坐标），协变＝**每个坐标各自的** `[x]` 量级。一个标量不可能同时逐坐标缩放。所以
「暖机协变」在逻辑上就等于「暖机不要护栏」。

三条已知的填洞路线，逐条看都不如放弃：

- **外部注入 `diag(H⁻¹)`**：二阶信息通常以整个 Hessian 约定，单独要一个逆 Hessian 对角是非常规
  接口，调用方（尤其取现成 benchmark）难以匹配；而能给出整个 Hessian 的调用方，本来就在用
  ADR-0001 明确拒绝的二阶路线。
- **拿边界盒宽 `(ub−lb)` 或 `|x₀|` 当逐坐标尺度**，做 `Δx = −η·w⊙sign(g)`：确实一阶且协变，但
  **盲步、无护栏**，陡函数上会冲过；无界问题还没有 `w`。用护栏换一步协变，不划算。
- **只在暖机改用二阶**：直接违反「仅依赖一阶信息」的前提（ADR-0001）。

而且这个洞的代价本来就接近零：需要协变的场景是光滑单盆地，那里 `d` 一步就读到 `1/H`、自动抹掉
暖机引入的尺度偏差；多峰场景协变本就不是重点。**故接受，且不再作为待办。**

### 1.5 停止条件

梯度范数 `‖g‖ ≤ τ_rel·‖g₀‖`（`τ_rel` 无量纲，`~1e-6`）。触发即该路径 `give_up`——它停下、不再
花 eval，把「要不要重来」的决定交给重启判定（第 3 节）。用范数而不用逐坐标 `|gᵢ| ≤ τ_rel·|g₀,ᵢ|`：
逐坐标的 all 语义会让「某个坐标在起点恰好为 0」的路径永不触发（伪存活），或让「某个坐标起点特别
大」的路径过早触发；范数只读聚合后的梯度尺度相对其起点，单条坐标不再单独卡住整条路径。这也是负
曲率/鞍点逃逸分支的挂点（本实现不接）。

## 2. 重启：样本自估计的 EI 判决（SOAR 式）

### 2.1 为什么不拟合代理模型

n_paths 条路径的起点是 n_paths 个 i.i.d. 抽样。**样本本身就是证据**：一次重启买到多少改进，可以直接用「重启
事件里那些起点的组最小值」来实测，而不必拟合 `f` 的代理模型。高维下拟合既贵又脆；采样估计的方差
只随样本数走，与维数无关。这就是 Mathesen et al. 2021 那个「自适应重启」框架的轻量版本。

### 2.2 两个 EI 闭式

都从同一份证据出发，闭式给出：

```text
EI_continue = max(B − (f_k + μ_p), 0)                    本路径继续一步的期望改进
EI_restart  = σ_r·[z·Φ(z) + φ(z)],  z = (B − μ_r)/σ_r     任一路径重来的期望改进
```

- `(μ_p, σ_p)` 是**本路径**当前 stage 内 `Δf` 的均值与总体标准差。stage = 距上次判决以来的若干步。
- `(μ_r, σ_r)` 是**共享 payoff pool** `{rᵢ}` 的均值与总体标准差。`rᵢ` 是每个选择事件（初始化或
  重启）里那些起点的组最小值——每次事件一次性把该事件的组最小值全追加进池子。

**池子是全路径共用一个，不是每条路径一份**：一次重启买到多少，是「抽样器 + 当前 incumbent」的
全局属性，与哪条路径无关。共用池子用最多样本估计同一个全局 `(μ_r, σ_r)`，方差最小；按路径切分只会
用更短的样本估同一个量。

### 2.3 判决与节奏：让两侧样本数可比

判决只在路径攒够一个 window 的 `Δf` 时问一次，问完就关掉这个 stage（下一个 `Δf` 开新 stage）。
`window = M`，`M` 是池子里可用的（有限）样本数。这个匹配是承重的：两边的相对标准差都约为
`1/√(2(n−1))`，只有样本数相当，`EI_restart` 与 `EI_continue` 才是可比的两个估计。若继续侧只读最近
两三步，它会低估自身抖动、把 `EI_continue` 压低，重启就会赢下证据并不支持的判决——这种偏差不会
报错，只会让最终 incumbent 更差。所以 window 是**推导出来的**，不是暴露给调用方的旋钮。

**不额外垫长度**：read 是 stage 上的等权均值与总体标准差，多折进来就是多一条证据、不多也不少；
唯一的例外是 `M = 0`（还没有任何有限 payoff）时窗口取 1——那不是先验，是「零样本的窗口读不出东西」。
`M = 1` 时两侧同时落到各自的退化形式（stage 一折无展布、池子一样本 `σ_r = 0`），此时任何一条重复
都赢不了 `EI_continue ≥ 0`，重启本就发不出来，所以也没有必要把窗口垫到 2。

判决谓词：`EI_restart > EI_continue` 且 incumbent 有限。非有限的 `Δf` 会留在 stage 里、但读的时候
跳过（一个 `+inf` 样本会让均值与标准差变 `NaN`，而 `NaN > EI_continue` 恒为 `False`，等于静默关掉
该路径的重启）。

### 2.4 n_paths=1 的退化

`n_paths = 1` 时池子里只有一个样本 ⇒ `σ_r = 0` ⇒ `EI_restart` 退化为 `max(B − μ_r, 0)`，与继续侧同形，
样本数也远不够打开判决 ⇒ **单路径永不重启**。这正是「把 n_paths 设成 1 就得到一条纯串行局部下降」的
退化路径，不需要任何模式开关。

## 3. n_paths 路径引擎

### 3.1 选择事件不盲降

初始化与重启都走同一个事件：抽 `n_probe_evals` 次求值、每次 `n` 个点（`n = n_paths` 或本次重启的路径数），
**按列**取最小值，得到 `n` 个起点；这些点的梯度直接复用（起点零额外 eval）。

为什么按列取最小：每一列跨求值就是从同一抽样器里抽出的一个随机组，组最小值既是「`n_probe_evals`
个随机点里最好的那个」（比盲降好），它的值又同时是 payoff pool 的一个**真实样本**——一次选择事件
买到的改进。所以「选起点」与「收集证据」是同一个动作，不额外花预算。`n_probe_evals = 1` 退化为
朴素初始化，`∞` 退化为随机搜索；默认 2。

### 3.2 每次迭代两步

```text
advance_paths(it) 对每条路径一次移动：先问判决（只在它的 stage 填满时真的问）并重启胜出的，
                  未重启且未 stalled 的路径再 propose → clip 进 box → 批量 serve → 逐路径 verdict → commit
observe_continue  每完成一次迭代喂一次 Δf（一次 step 一个样本）
```

判决与步进合成一个函数是刻意的：一次迭代只推进「一步」，判决读的是本轮自己的 B；被拒的探测
跳过本轮步进，并在请求点抛出私有异常 `_BudgetExhausted`，径直 unwind 到最外层循环的 catch。

重启的路径本轮只做探测、不进 `step_paths`：一条路径一次迭代要么探测要么步进，不会两头都花预算。

`observe_continue` 的样本就是**一次迭代的一步**：每条路径各自计时、互不重叠。重启的路径本轮只探测、
不步进，它这一次迭代不喂样本（重启把锚点重置到探测点的 f，第一个 Δ 留给下一步），所以既不会把探测
的收益记成一步，也不会把探测读成零步。任意一次迭代 stalled 就喂 `inf`——宁可丢一个样本，也不让横跨
停顿的 Δ 冒充进步。

判决两侧买的东西并不相等：一次探测花 `n_probe_evals` 个批、算 `n_probe_evals` 次梯度，而一步只算
一次梯度（默认 `n_alpha=1`）。真正吃重的是**梯度次数**（g 约为 f 的 2–5 倍），所以 `n_probe_evals > 1`
在不接 f-only 求值器时就偏向探测这一侧；默认值因此是 1，接上 f-only 后两侧各只算一次梯度，剩下的只是
便宜的 f 计数差（完整推导见 `docs/PLAN.md`）。

### 3.3 单一 argmin 所有者与 `fold`

一次 step 里若某方法扫了一个步长网格，引擎把所有被服务的 trial 中 `f` 最低的那个提交。**引擎是
「谁赢」的唯一所有者**：方法只通过 `fold` 回调得知获胜点，不自己维护第二份 argmin。这样重启/折叠
的簿记不会与方法的内部状态漂移。

### 3.4 `stalled` 是唯一的路径状态

一条路径在某一次迭代「没有任何移动」（既没重启、也没提交步）就叫 stalled——可能是自己的收敛
判据 `give_up`，也可能是重试次数 `max_ls` 耗尽。stalled 路径和本轮刚重启的路径都不进
`step_paths`，但每个 stalled 的迭代仍然喂一个 inf 的 `Δf`（让它的判决节奏与同批前进的路径对齐，
而不是领先一个样本），直到某次 EI 判决把它唤醒。

### 3.5 预算计在批边界（引擎自己记账）

`max_evals` 是引擎自己的预算，单位是**批**：服务一批就记一笔，所以一次探测（`n_probe_evals` 批，
初始化探测也算）自己就吃掉 `n_probe_evals`——探测不是免费的（ADR-0004）。记账在批之前问：要么整批
服务、要么整批不做，不做部分前缀。`_Budget` 的两个计数器是它全部的状态：`charge` 只记账，`accept`
是唯一的停车查询（批封顶或调用方的闸关闭），拒绝不存成状态——否则它就和自己的证据脱钩了。同一个
数字也是循环头的停车条件：一次完成的迭代可能刚好把批数花到 `max_evals`（或把闸关上），所以在开
新一轮之前先查一次，不再为用不掉的预算问判决；循环头这一处问的是「要不要开这轮」，属于决策，
不是传信号。迭代中途被拒（批封顶或 `budget_check`）则在请求点抛 `_BudgetExhausted` 并结束，该迭代
不计入 `nit`：中间层（`probe_select`、`apply_restarts`、`step_paths`）因此不必再检查返回值、也
不必上下传任何标志。停滞的迭代不服务任何东西，也就不记账。

`budget_check` 只留给引擎看不见的预算（点数、墙钟、钱）。它同样在批边界翻转，所以那类外部预算可能
超跑不到一个批大小；`max_evals` 本身在批上是精确的，一个方法不会比另一个多拿一批。

### 3.6 全停早退

所有路径都 stalled、且下一张判决票按 `samples_until_verdict()` 折算成迭代数（一次迭代一个样本，所以
样本数就是迭代数）也落在剩余迭代数（`max_evals` 减去已完成迭代数）之外时，引擎直接结束
（`all_stalled`），不再空转——没有 eval 会再发生，`x`/`f`/`n_points` 不会再变。首次样本落账前
`needed` 为 0，`0 >= 剩余数` 为假，捷径暂不生效；但之后 `needed >= 1` 且剩余数单调减小，
必定在 `it` 触到 `max_evals` 之前触发，所以循环不需要额外的「迭代数上界」。

## 4. 适用界

- 这是**局部**方法 + 采样重启，不是全局搜索器；它靠 n_paths 条路径的多样性与重启频率换全局性，收敛
  保证与 Adam/RMSProp 同级，不追牛顿/拟牛顿。
- 逐坐标估计忽略耦合项 `Σ_{j≠i} Hᵢⱼ sⱼ`，后者落在 `|yᵢ|` 内使 `d` 成为 `1/Hᵢᵢ` 的有偏估计
  （耦合越强偏差越大，倾向系统性欠步：更稳但更慢）。这是拿「安全」换「潜在快」。
- 重启频率与 `n_probe_evals`、`β`、`θ`、`τ_rel` 相关；文档只给结构，不给某类问题的推荐值——
  那些值由目标决定。

## 5. 两条成本轴（成本模型，不是评分）

一次求值服务一批 `(M, d)` 的点，所以同一次运行有两个代价数，它们回答不同的问题，**永远不相等**：

| 字段 | 数什么 | 回答的问题 | 用途 |
| --- | --- | --- | --- |
| `n_points` | 被服务的**点**（一批 n_paths 点计 n_paths） | 串行效率：一个点买到多少改进 | 串行求值（逐点计费）的计分/门控口径 |
| `n_batches` | 被服务的**批**（一批 n_paths 点计 1），一次求值一次 | 并行效率：一次调用买到多少改进 | 批量目标（真 vmap/GPU）的代理口径 |

两个都要报：串行轴下，n_paths 条路径的收益会被 n_paths 除，多路径的账面上限被压低；并行轴下一次求值
就是一次并行调用的成本，「多样本 + 自适应重启」的收益才如实显形。两者都不等于墙钟（换底层语言能把
固定开销压掉，但改变不了「一次求值要服务几个点」这个结构量）。

预算只记在批上（3.5），所以 `max_evals` 与 `n_batches` 同轴；`n_points` 只报不预算——引擎无法在一个
不可分的批内部设卡。**不许混写**：同一张表、同一张图只能用一条轴，且必须标注。把 `n_batches` 的数字
当成 `n_points` 的数字去比较，等于把 n_paths 倍的预算差抹平。

## 6. 可换的实现选择 vs 不许破的不变式

参考实现把接缝切成四块：局部方法（`init/restart/propose/verdict/fold`）、路径点与 verdict 类型、
批量目标 callable、重启规则；另外允许注入起点抽样器。**这些签名全是实现选择**，新项目可以：

- 把局部方法写成纯函数 + 显式状态（不需要协议类）；
- 把规则写成闭包或普通函数（不需要一个 `_stage_window()` 方法）；
- 把批量目标写成任何可 vmap 的 callable，甚至写成一次同步阻塞的远程调用；
- 把 `PathPoint`/`Verdict` 换成元组/枚举/字符串——只要语义一致。

#### 必须保留的不变式（换了就坏）

1. `d ≥ 0` ⇒ `−d⊙g` 恒下降 ⇒ 稳态不需要 acceptance test（1.2 之 1）。
2. 单位不变（1.2 之 2），以及旋钮全无量纲（1.2 之 3）。
3. 地板用 `max`，λ 只在 `Y < λ` 时生效（1.3）。
4. 暖机有护栏、稳态无护栏，理由是有没有 `d`（1.4）。
5. 判决两侧样本数可比（`window = M`，空池取 1），池子全路径共享（2.3）。
6. 单一 argmin 所有者：引擎决定谁赢，方法只通过 fold 得知（3.3）。
7. 预算门控在批边界，不切分批次（3.5）。

## 7. 测试带走清单

按「测什么」分三层。前两层随算法走，第三层是本仓边界、不带走。括号里是参考实现里的函数名——它们随这些
测试一起搬走，所以在新项目里同样可解析；换名字无所谓，语义在就行。

#### 算法核（跟着局部方法走，`test_solver.py`）

- 地板是 floor 而不是加数：`test_diag_floor_is_a_floor_not_an_addend`（构造 `Y = λ` 的交叉 secant，
  断言 `max` 形式、排除加法的 `1/√2` 因子）。
- 两个旋钮是真的：`test_diag_armijo_c_is_a_knob`（`c=1e-4` 接受、`c=0.5` 拒绝同一 trial）、
  `test_diag_warmup_is_a_monotone_backtracked_gradient_step`（回溯序列随 `backtrack` 走）。
- secant/幅值比：`test_diag_l2_step_is_newton_exact_on_a_quadratic`、
  `test_diag_abs_step_is_newton_exact_on_a_quadratic`（二次函数上 `d = 1/h`，一步到 0）。
- EMA 窗口：`test_diag_decay_blends_later_secants`。
- 地板只守 `Y→0`：`test_diag_gradient_floor_carries_a_flat_secant`（`y = 0`、`g ≠ 0` 仍有有限下降步）。
- 单位不变：`test_diag_step_scales_with_the_coordinate_unit`（同一 secant 在缩放坐标下 `d̃ = c²d`、
  `Δx̃ = c·Δx`；注意要注入 secant 对，不能走暖机——暖机步按设计就不是坐标不变的）。
- 停止条件：`test_diag_gives_up_only_when_the_gradient_norm_converged`、
  `test_diag_gives_up_on_a_nonfinite_trial`、`test_diag_gives_up_on_a_nonfinite_gradient`。
- 暖机重试非有限 trial：`test_diag_warmup_retries_a_nonfinite_trial`。

#### 引擎（跟着 n_paths 路径框架走，`test_solver_seam.py` / `test_scheduler.py` / `test_restart.py`）

- 接缝真的被驱动（而不是走默认）：`test_injected_solver_is_driven`、`test_injected_rule_is_driven`。
- 选择事件不盲降 + 批形状：`test_batch_shape_and_point_counting`、
  `test_probe_group_min_skips_nan`、`test_probe_inf_group_min_does_not_silence_restarts`。
- `stalled` 语义：`test_give_up_stalls_a_path`、`test_stalled_paths_wait_for_ei_then_restart`、
  `test_nonfinite_start_waits_for_ei_instead_of_stepping`。
- 单一 argmin 所有者：`test_diag_line_search_walks_the_grid_and_folds_the_argmin`、
  `test_diag_line_search_gives_up_when_no_sample_is_finite`。
- 判决节奏与可比样本数：`test_window_is_the_pool_size`、
  `test_the_two_sides_read_the_same_samples_the_same_way`、
  `test_verdict_is_due_once_per_window_and_closes_the_stage`、
  `test_samples_until_verdict_counts_to_the_next_full_stage`。
- 池子共享 / stage 独立 / 重启保留池子：`test_payoff_pool_is_shared_across_paths`、
  `test_per_path_stage_is_independent`、`test_restart_keeps_the_shared_payoff_pool`。
- `n_paths = 1` 退化（`σ_r = 0` ⇒ 永不重启）：`test_no_restart_evidence_never_restarts` 覆盖闭式退化，
  `test_no_incumbent_never_restarts_but_does_not_stall_the_clock` 覆盖 B 缺失；**没有端到端 n_paths=1 断言**
  ——搬走前可补一条（n_paths=1 跑满 max_evals，断言 `restarts == []`）。
- EI 闭式：`test_normal_functions_match_known_values`、`test_ei_continue_exact_degenerate_at_sigma_zero`、
  `test_ei_continue_matches_gaussian_ei_literal`、`test_ei_restart_matches_gaussian_fit_literal`、
  `test_ei_restart_fits_a_population_std`、`test_ei_restart_gaussian_tail_below_all_observations`、
  `test_ei_restart_empty_single_and_nonfinite_skipped`。
- **两条成本轴**：`test_reports_points_and_batches_separately`（`n_batches` 与 `n_paths` 无关，
  `n_points` 等于 `n_batches × n_paths`）。
- 预算计在批边界：`test_budget_exhaustion_stops_everything`、`test_no_budget_run_reports_max_evals`、
  `test_budget_is_never_exceeded_under_restart_storm`（探测记账：重启风暴也超不过 `max_evals` 批）、
  `test_every_path_feeds_once_per_iteration`（每迭代恰好一个样本，停顿的那个喂 inf）。
- 盒是数据：`test_box_is_data_not_a_unit_cube_assumption`、`test_bounds_spellings_are_the_same_box`、
  `test_unbounded_or_absent_box_needs_an_injected_sampler`、
  `test_the_sampler_is_the_only_authority_on_d`、`test_an_injected_sampler_searches_outside_the_old_box`。

#### 本仓边界（不带走）

`test_shell.py` / `test_shell_seam.py` / `test_submission.py`（dfbench shell 接缝、official-validator
冒烟）、数值指纹逐位比对、提交归档静态校验。这些依赖 dfbench 或交付包，与算法无关。

## 8. 与 SOAR 的关系

名字里的 "soar" 向 Mathesen et al. 2021 的**自适应重启**框架致敬，借的只是那一条思想：重启能买到的
改进不必拟合 `f` 的代理模型，而用抽样的起点组最小值**实测**（第 2.1 节）——后缀 `-lite` 正指「只保留
这一条的最小实现」。其余部分是两回事：

- SOAR 是一个**全局–局部协同的优化框架**（文献含代理/集成模型与统计接受检验等成分）；本包不拟合
  任何 `f` 的模型、没有多保真，只有一个局部方法 + 一个重启判定。
- 重启判据不是 SOAR 的统计接受检验，而是共享 payoff pool 上 `EI_restart > EI_continue` 的闭式比较
  （第 2.3 节）。
- 局部方法默认是逐坐标 secant 幅值比预条件（第 1 节），一阶，无 Hessian、无曲率模型。

本包**不是** SOAR 的移植、子集或简化实现，引用它仅为归属与差异定位（第 2.1 节正文）；不暗示接口
兼容或结果可复现。

## 9. 参考文献

- Conn, A. R., Gould, N. I. M., & Toint, P. L. (2000). *Trust-Region Methods*. SIAM/MPS.
  （割线方程 `B s = y` 与 SR1 更新；后者的对角退化就是本文的比值）
- Mathesen, L., Pedrielli, G., Ng, S. H., & Zabinsky, Z. B. (2021). Stochastic optimization with
  adaptive restart: a framework for integrated local and global learning. *Journal of Global
  Optimization*, 79, 87–110.（SOAR 式样本重启）
- Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive
  black-box functions. *Journal of Global Optimization*, 13, 455–492.（EI 闭式）
