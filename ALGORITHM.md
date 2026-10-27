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

1. **恒为下降方向**：`d ≥ 0` 由构造保证（绝对值/平方/开方 + 正地板），所以 `(−d⊙g)·(−g) ≥ 0` 逐坐标成立。
   预条件后仍是下降方向 ⇒ **`−d⊙g` 的步不需要任何 acceptance test**，不需要线搜索去「验证方向」。
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

两者在 `Y > 0`、`λ > 0` 时都不相等，只在 `Y→0` 与 `Y→∞` 两个极限趋于同义。全域关系是
`d_add = d_max·√(max(Y,λ)/(Y+λ))`（`"rms"`；`"abs"` 不带根号）：加法拼法给出的 `d` 恒小于 `max`
拼法——**λ 在每个 `Y` 上都在压低步长，而不只在触底时**。
比值 `d_max/d_add = √((Y+λ)/max(Y,λ))` 在 `Y = λ` 处取峰值 `√2`（`"abs"` 因未开方，峰值是 `2`），
`Y→0` 与 `Y→∞` 两端都趋于 1。也就是说加法地板把 `λ` 混进了「步长调节」的职责，而你无法再由 `Y`
读出「这个坐标是不是曲率主导」。

`max` 把职责分开：**λ 只做 `Y→0` 的守卫**，`Y` 一旦高于 `λ`，`d` 就忠实等于 `S/Y`。步长该多激进
由独立旋钮承担（`alpha0` 全步阻尼、`backtrack` 网格间距、`n_alpha` 网格样本数），不塞进地板里。这不是凭经验调出来的
收益，而是把估计量从「被 λ 污染的比值」还原成「比值本身」——解耦，而不是调参。

### 1.4 暖机步：同一条 alpha 网格，只是方向还没有 `d`

暖机与稳态共用**同一种步机制**：沿自己的方向采样 `alpha` 网格
`{alpha0, alpha0·backtrack, alpha0·backtrack², …}`，引擎的逐步 argmin 决定谁赢，`fold` 记录赢家。
两相的差别只在**方向**——第一步还没有 secant 对，`d` 无从估计，只能沿 `−g₀`；拿到首个
`(s, y)` 之后换成 `−d⊙g`。方向之外的一切（`alpha0` 起点、`backtrack` 间距、`n_alpha` 样本数）两相同用。

默认 `n_alpha = 1` 时两相都是**一次无护栏的整步**：暖机提交 `x − α0·g₀`，稳态提交 `x − α0·d⊙g`，
都不读值、不回溯、不与起点比较。`needs_verdict(i)` 因此**与相位无关**：`n_alpha > 1` 答 True，
否则答 False（提交即所提，见 3.2）。

#### 为什么暖机不再需要（也不再享有）护栏

旧设计给暖机一条单调 Armijo 回溯，理由是「第一步没有 `d`，所以要有护栏」。但那道护栏实际保护的是
**`f` 值**，不是 secant 信号：二次函数上 `d = s/y = 1/h` 与步长无关，超调一步之后紧接着的那个 secant
仍然精确（`test_diag_l2_step_is_newton_exact_on_a_quadratic` 正是这样证的，超调 25 倍仍一步到 0）。
于是在已测的两个地貌上，护栏只付代价：

- **对角二次形**：`alpha0 = 1` 本来就是 one-shot 最快，回溯把一次超调拖成十几次 eval。
- **旋转二次形**（`cond = 1e6`）：误差在**方向**（对角 secant 丢掉非对角曲率），标量 `α` 结构性修不了；
  回溯在那个域里不改变结局。
- **单位重标定残差**：`experiments/DESIGN.md` 量到的 1.6× 点数差**整个**由暖机回溯次数造成
  （base 12 次 reject、重标定 5 次，恰好差 `log2(s²)`）。删掉回溯后这个可见残差按构造消失。

代价是失去「首对 secant 的自动局部化」：非二次、且 `alpha0` 尺度失调时，一次盲整步种下的 `1/H`
不如小步局部。这个损失**从未被测过**（`DESIGN.md` 明标 unmeasured），而且它有三条出路，都不用再引入
第二种机制：超调自校正（`y` 大 ⇒ `d` 小）、`alpha0 < 1` 手动阻尼、`n_alpha >= 2` 的网格重新局部化——
与稳态同一套旋钮。

#### 暖机步仍不协变，这一步不修

删掉护栏**没有**修掉方向问题：`−g₀` 是**梯度类量**（`x̃ = c·x ⇒ g̃ = g/c`，坐标放大、梯度反而缩小），
把它乘上标量 `α` 塞进 `x + α·(−g)`，步长仍随坐标刻度反向缩放；稳态的 `−d⊙g` 才是**位移类量**
（`d` 已把 `1/H` 的量纲补进去）。三条已知填洞路线都不如放弃：外部注入 `diag(H⁻¹)`（非常规接口，
且能给出整个 Hessian 的调用方本就在用 ADR-0001 拒绝的二阶路线）、拿边界盒宽 `(ub−lb)` 或 `|x₀|`
当逐坐标尺度做 `Δx = −η·w⊙sign(g)`（一阶且协变，但盲步无护栏，无界问题还没有 `w`）、只在暖机改用
二阶（违反「仅依赖一阶信息」的前提）。

代价本来就接近零：需要协变的场景是光滑单盆地，那里 `d` 一步就读到 `1/H`、自动抹掉暖机引入的尺度
偏差；多峰场景协变本就不是重点。**故接受，且不再作为待办。**

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
EI(f⋆; x, h) = h·[z·Φ(z) + φ(z)],  z = (f⋆ − x)/h        单核高斯 EI（h = 0 → max(f⋆ − x, 0)）
h(样本)      = 1.06 · MAD(样本)/Φ⁻¹(3/4) · n^(−1/5)    scott_mad 带宽（1.06 = (4/3)^(1/5)）
EI_continue  = mean_j EI(f⋆; f_k + Δⱼ, h_Δ)              本路径继续一步的期望改进
EI_restart   = mean_i EI(f⋆; mᵢ, h_m)                   任一路径重来的期望改进
```

- `Δⱼ` 是**本路径**当前 stage 内的一步变化。stage = 距上次判决以来的若干步。
- `{mᵢ}` 是**共享 group-min pool**：`mᵢ` 是每个选择事件（初始化或重启）里那些起点的组最小值——每次
  事件一次性把该事件的组最小值全追加进池子。
- 两侧**都不假设样本服从单个高斯**：各自用样本本身做高斯核 KDE（一折一核），偏斜的 `Δf` 流也照原
  样读。带宽取稳健的 MAD 尺度而不是标准差，所以一条长尾顶不爆它。

**池子是全路径共用一个，不是每条路径一份**：一次重启买到多少，是「抽样器 + 当前最优 `f⋆`」的
全局属性，与哪条路径无关。共用池子用最多样本估计同一个全局核密度，方差最小；按路径切分只会
用更短的样本估同一个量。

### 2.3 判决与节奏：让两侧样本数可比

判决只在路径攒够一个 window 的轮次时问一次，问完就关掉这个 stage（下一个 `Δf` 开新 stage）。
`window = M + 1`，`M` 是池子里的**样本数**（池子多长就是多少：`+inf`、`nan`、`-inf` 也都是样本）；
那个 `+ 1` 是 stage 开张的那一轮——它对继续路径是这次判决本身、对重启路径是那次探测——任何 stage
都要花掉它却不折到样本。这个匹配是承重的：两侧是同一种 scott_mad KDE，
只有样本数相当，它们的带宽与核均值才在同一个精度上，`EI_restart` 与 `EI_continue` 才是可比的两个
估计。若继续侧只读最近两三步，它会低估自身展布、把 `EI_continue` 压低，重启就会赢下证据并不支持
的判决——这种偏差不会报错，只会让最终的 `f⋆` 更差。所以 window 是**推导出来的**，不是暴露给
调用方的旋钮。

**两侧样本数必须相等，多出来的那一个在继续侧**：stage 开张的那一轮，继续路径折进一个样本、重启
路径只做一次探测，所以继续路径到下一次判决时手里有 `M + 1` 个 `Δ`。读它的**最近 M 个**（丢掉最早
那个）是让两侧数目对齐的动作，而不是额外等一轮：window 已经把开张那一轮算进去了，再等一轮才是
白等。

**不额外垫长度**：read 是 KDE 上的等权、一折一核，多折进来就是多一条证据、不多也不少；
窗口不垫到 2（`M = 1` 的退化形式见下）。空池子的 window 是 1，但那条路径由冷启动直接放行、
永远读不到它。
停摆路径根本不读 stage：它的继续侧按定义为 0，所以判决退化成单侧 `EI_restart >= 0`——`EI_restart`
非负，于是这条比较恒真，停摆路径每个窗口到期就重探一次（平局条款）。到期只由一条
**阶段钟**决定——它数的是轮次而不是折数：引擎对每条路径每轮喂一次 `observe_step`，移动的报 Δ、
折叠什么都没发生的轮报 `None`（停摆，或重启路径的探测轮）——探测轮既没走一步也没停摆，就按无步轮报。
一条通道保证每轮每条路径恰好走一格，所以一条路径的判决轮次与它这一程怎么走来无关。中途停摆的路径因此保留它已经等过的轮数，
不会从停摆那一刻重新起算（否则它会比同 cohort 晚整整一个窗口才到期）；变的只是证据从全比较
变成单侧。重启路径也一样：少了那一格，它每重启一次就把自己的下一程推后一轮、并与同 cohort
永久错开（锁相）。
`M = 1` 时两侧同时落到各自的退化形式（stage 一折、池子一样本，MAD = 0 ⇒ `h = 0`）：唯一的样本已是
`f⋆`，所以 `EI_restart` 恒为 0；继续侧走平或停摆时也读 0，判决以 `0 ≥ 0` 平局收场、放行
一次探测——这一次探测正是池子买到第二个样本的方式。窗口因此不需要垫到 2。

判决谓词：`EI_restart ≥ EI_continue`；`f⋆` 非有限时不经比较、直接重启（见下）。取等即平局也重启：
两侧同为 0 意味着「继续买不到东西、重启也买不到」，而一次探测本身还是一条真实的 i.i.d.
group-min，所以平局时探测是唯一能买到证据的动作。它不会空转：窗口就是池子长度 M，每次探测都往
池子追加一个样本，同一条路径的下一次问询因此推迟整整一个窗口，间隔按 1, 2, 3, … 增长，探测
频率衰减到 0。`f⋆` 非有限（`+inf`：至今没服务过任何有限点）时没有可比的量，判决回到冷启动那条
原则——**无法比较时，探测是唯一能买到证据的动作**——所以到期就重启：否则唯一能重新抽样的那条
路径恰好被「要先有有限样本」这一步挡住，一次不走运的初始化探测就能把整个 run 锁到地平线。这里的
`f⋆` 是**全域**最优 `fun_raw`
（所有有限样本、含不可行点），与 `f_k` 和池子里的 payoff 同口径；结果里报告的 `fun`/`x` 仍是
可行域最优。池子里的 `+inf`、`nan`、`-inf` 都是**零项**，不是被丢掉的值：它们对改进的贡献恰好是 0
——`+inf` 是死起步，买不到东西；`nan`/`-inf` 是有限集之外的事件，和 `+inf` 一样不说明重启能买到多少——
所以它们作为同一个均值的零项进账：`EI_restart` 等于有限子集的 KDE 均值乘以「有限样本占比」
`|finite| / M`，带宽只在有限子集上算（零项不许把它撑开）。这样重启侧读的也是同一个 M 里的均值，
全死池子无法伪装成一个更小、更尖、因而更贵的池子。

### 2.4 n_paths=1 的退化

`n_paths = 1` 时池子里只有一个样本 ⇒ 无展布 ⇒ `h = 0` ⇒ `EI_restart` 退化为 `max(f⋆ − r, 0)`，而池子
里的那个样本正是 `f⋆`（最优必不劣于唯一抽样），所以重启侧恒为 0。判决于是等价于「继续侧是否为
0」：路径在改进时继续侧为正、单路径就一直下降；走平或停摆时平局、放行一次探测去补第二个样本。
**要一条纯粹的串行局部下降、一次都不重启，传入 `NoRestartRule`**——纯串行不再依赖 `n_paths = 1`
的偶然：多起点的纯下降实验（对比重启机制）也走同一个显式规则。

## 3. n_paths 路径引擎

### 3.1 选择事件不盲降

初始化与重启都走同一个事件：抽 `n_draws` 批点、每批 `n` 个（`n = n_paths` 或本次重启的路径数），
**按列**取最小值，得到 `n` 个起点；这些点的梯度直接复用（起点零额外 eval）。

为什么按列取最小：每一列跨批就是从同一抽样器里抽出的一个随机组，组最小值既是「`n_draws`
个随机点里最好的那个」（比盲降好），它的值又同时是 group-min pool 的一个**真实样本**——一次选择事件
买到的改进。所以「选起点」与「收集证据」是同一个动作，不额外花预算。`n_draws = 1` 退化为
朴素初始化，`∞` 退化为随机搜索；默认 1。

### 3.2 每次迭代：探测半与步进半

判决与步进同在一轮迭代里是刻意的：一次迭代只推进「一步」，判决读的是本轮自己的 `f⋆`；被拒的探测
跳过本轮步进，并在请求点抛出私有异常 `_BudgetExhausted`，径直 unwind 到最外层循环的 catch。

重启的路径本轮只做探测、不提交步：一条路径一次迭代要么探测要么步进，不会两头都花预算。两半碰的
是互不相交的路径，所以「先定完再一起买 g」与「逐个定完就买」是同一个日程。

`observe_step` 的样本就是**一次迭代的一步**：每条路径各自计时、互不重叠。重启的路径本轮只探测、
不步进，它这一次迭代报一个无步轮 `None`、不喂样本（重启把锚点重置到探测点的 f，第一个 Δ 留给下一步），
所以既不会把探测的收益记成一步，也不会把探测读成零步。任意一次迭代没有折（stall）也报 `None`——那是
缺席不是值：没有 Δ 可读，但这一轮照样计入阶段钟，所以停摆不会把判决推迟，也不会拿一个横跨停顿的 Δ
冒充进步。

判决两侧买的东西并不相等，而且**接不接 f-only 接缝会改变这笔账**。真正吃重的是**梯度次数**（g 约为
f 的 2–5 倍），所以问题总是「这一侧要买几次 g」。两个旋钮的默认值都是 1：`n_draws = 1`（一次探测一抽）、
`n_alpha = 1`（一步一个候选）。

- **不接接缝**：每一抽、每个 trial 都是融合批（值与梯度一起送达），所以一次探测付 `n_draws` 次梯度、
  一个判定步付 `n_alpha` 次梯度——两侧各由自己的旋钮乘上去，平不平要看两个旋钮之比。引擎核对不了这个
  比值：它只认 `LocalSolver` 这层契约，一个判定步要花几次梯度是方法自己的选择（`verdict` 契约原文），
  `n_alpha` 只是具体方法（如预条件方法）的私有成员。它能表达的保守代理就是「没接接缝 ⇒ 探测不比一次
  梯度贵」，于是 `n_draws > 1` 抛 `ValueError`；默认的 `n_draws = n_alpha = 1` 恰好让两侧各付一次梯度。
- **接了接缝**：搜索期**只读需要读的值**——探测的抽与**要判定的**步的每个 trial 都只服务 f，判决也只听
  这些 f（3.3），留到最后的点（每列的探测起点、每条路径步的赢家）在同一批里补 g。于是两侧都恰好一次
  梯度往返，与 `n_draws`、`n_alpha` 都无关：延迟是接缝的性质，不是抽数的性质。

接缝下多付的只有 f，而单抽连这份都不付——它没有组内可比的候选，起点不必先读值，随补 g 的那一批一起
求 f 与 g（多抽才需要先把每抽的 f 读出来做 group-min，那次 f 随后被融合批重算，这是用便宜的一半换掉
一次梯度往返）。完整推导见 `docs/PLAN.md`。

一步只有在方法**要判定**时才读值：`needs_verdict(i)` 一次问清这条路径下一步是否逐候选判定——`n_alpha > 1`
的步长网格（暖机与稳态同一规则）答 True（每个样本决定下一次移动），默认的整步答 False（提交即所提）。
答 False 的那一步与单抽探测走同一条延迟通道：引擎不为它服务 f，把提议当作待补点交给 `settle`，值与
梯度在同一批里各求一次。所以「一步的 f 算一次还是两次」由方法的判定需求决定，不由接缝或抽数决定：
有 `eval_f` 时，被判定的步付一份 f-only 读值再在融合批里重算同一个 f，不判定的步只付融合批那一次。

探测点要能在优化里用，`f` 和 `g` **两半都得有限**才算数：`f` 有限但梯度非有限的起点会被 `propose`
读取（`-p.g`）并进 `fold`（`best.g - p.g`），一个 NaN 就能把那个坐标的量级 EMA 永久毒死。这样的
起点按「死抽」处理、到达即 stalled，等 EI 判决复活。

### 3.3 单一 argmin 所有者与 `fold`

一次 step 里若某方法扫了一个步长网格，引擎把所有被服务的 trial 中 `f` 最低的那个提交。**引擎是
「谁赢」的唯一所有者**：方法只通过 `fold` 回调得知获胜点，不自己维护第二份 argmin。这样重启/折叠
的簿记不会与方法的内部状态漂移。

方法的判定因此只收**坐标与值**（`verdict(i, p, x, f)`：`p` 是这一步的起点，`x` 是夹进 `bounds`
后实际服务的那一点）：trial 的梯度刻意不在契约里，因为引擎只为赢家买 g，方法拿不到别的 trial 的
梯度；需要 trial 梯度才能判决的方法不是这个引擎能驱动的对象。相应地，赢家的梯度是**在提交点才
第一次被看见**的（`f` 先定，g 随收）；不判定的步连 `f` 也是这时才第一次看见（3.2）。所以提交闸
对**两半**都查：`f` 或 `g` 任一非有限就不提交、按 stalled 处理（与死抽同路）。这样 `fold` 是唯一的
提交点，方法不会先被告知「落点」再被收回，非有限值也进不了规则的下一个 Δ 与 EI 的池子。

### 3.4 `stalled` 是唯一的路径状态

一条路径在某一次迭代「没有任何移动」（既没重启、也没提交步）就叫 stalled——判据只有方法的
`give_up`：枚举多长是方法自己的事（它的收敛判据、它自己的试用预算），引擎**没有试用上限**。把上限
留在引擎侧就是给方法已经拥有的决定加第二个隐形停止规则；代价是一项契约：`verdict` 必须最终返回
`accept` 或 `give_up`，永不终止的枚举是挂死而不是慢跑。stalled 路径和本轮刚重启的路径都不进
`step`，但每个 stalled 的迭代仍然报一次 `None`（让阶段钟照走，而不是冻结在停摆那一刻），
直到某次 EI 判决把它唤醒。停摆的继续侧按定义为 0——继续买不到东西——所以它的判决退化成单侧
`EI_restart >= 0`：`EI_restart` 本身非负，这条比较恒真，于是停摆路径在每个窗口到期时重探一次（2.3 的
平局条款）。**到期即结阶段**：判决无论读成什么，`deltas` 与阶段钟都在判决点归零（一个窗口换一个阶段，
阶段何时结束不看答案）；于是回 False 之后要再攒一个窗口才重问，而不是每轮重问同一个问题——既不是
探测风暴，也不是每轮一次的空判。（停摆路径不存在「判决一直答 False」的停摆：到期要么平局要么没有
可比的 `f⋆`，两者都重启；真正空转的情形见 3.6。）

### 3.5 停车是三闸门，记账只报告

`max_iters` 是引擎自己的地平线，单位是**迭代**：一次完成的迭代记一笔（迭代 0 就是初始化探测）。
它是确定性的停车界，不是成本代理——一次迭代服务多少批是方法的自由，从零批（全停滞的一轮）到一次
探测加上方法自己线搜索的若干试用批。`time_limit` 是同一件事的墙钟版本，给那些「一次迭代多贵说不准」
的调用方。
两者至少要给一个，否则入口抛 `ValueError`：只有 `budget_check` 停不住空转的羊群（停滞的迭代不服务
任何批，也就永远不会被批级闸拦住）。

`_Budget` 因此合了两个职责：停车计量（`max_iters`、`time_limit`、`budget_check`）与工作量报告
（`n_batches`、`n_points`）。**只有 value+grad 的融合批（g 批）计入账单**：探测的 `n_draws` 个
只有值的批便宜，既不记账也不问闸，所以 `n_batches`/`n_points` 正好落在承重轴（梯度次数）上。
`charge` 只记账，`tick` 只在迭代体完成后记一次迭代；`accept` 是唯一的
停车查询（地平线到了、墙钟到了，或调用方的闸关闭），拒绝不存成状态——否则它就和自己的证据脱钩了。
同一个数字也是循环头的停车条件：一次完成的迭代可能刚好把地平线走到（或把闸关上），所以在开新一轮
之前先查一次，不再为用不掉的预算问判决；循环头这一处问的是「要不要开这轮」，属于决策，不是传
信号。迭代中途被拒则在请求点抛 `_BudgetExhausted` 并结束，该迭代不计入 `n_iters`：中间层
（`probe_select`、`restart`、`step`）因此不必再检查返回值、也不必上下传任何标志。
`status` 按 `max_iters` → `time_limit` → `budget_exhausted` 的顺序报第一个成立的闸。另一种停车是
`unbounded`：某个 **feasible** 点把目标值打到 `-inf`，那里没有下界，任何后续试用都比不过它、任何一步也改不动它，所以该点直接成为答案（`fun = -inf`，`x` 就是它）并结束运行。这是引擎唯一自己能认出的停车条件，
而且它仍然读 feasibility 掩码：不可行的 `-inf` 不改变任何东西（既不进 `fun`，也不进 `fun_raw`）。
找到它的那一批已经服务并记账（`n_points`/`n_batches` 计入），但这一轮没跑完，因此和批被拒一样不入账
`n_iters`。

`budget_check` 只留给引擎看不见的预算（点数、墙钟、钱、外部早停策略）。它同样在批边界翻转，所以那类
外部预算可能超跑不到一个批大小。它只在 g 批前被问：f-only 段不查，因此那一段的停车
（含墙钟）粒度是「不到一个探测/一步」——接了接缝时一次迭代最多只有一个 g 批（3.2 的那个合并批），
它同时被这道闸计数。这是一个有意接受的代价，靠「f 便宜、g 约为 f 的 2–5 倍」的前提兜底；f 本身贵的
调用方就不该接 f-only 接缝、也不该开多抽探测。`n_batches`/`n_points` 只报不预算：
引擎无法在一个不可分的批内部设卡。

### 3.6 全停不早退：空转到下一次判决，或者空转到地平线

所有路径都 stalled 时，这一轮不服务任何批，但每条路径仍然报一次 `None`（3.4），阶段钟照走。之后去
哪里，看那条路径到期时有没有判决可给：

- **到期就判（本包的 `EIRestartRule`）**：判决恒真——`f⋆` 有限时是单侧 `EI_restart >= 0`，
  `f⋆` 非有限时直接重启（2.3），于是每个窗口到期就重探一次、这一轮就有批可服务：空转只在窗口与
  窗口之间，不贯穿整段。重探的间隔随池子变长（2.3），所以一停到底的羊群变成的是一串越来越稀的
  探测，不是探测风暴。实测（2 路径、`max_iters=40`、求值永远 NaN）：6 次探测、6 批，不是满额 40 批。
- **永不让判决到期**：第三方规则可以把窗口设得比地平线还长（`test_stalled_paths_wait_for_ei_then_restart`
  的对照臂就是这种），那样重启一次都不发生，引擎一路空转到 `max_iters`/`time_limit` 停（`status`
  报 `max_iters` 或 `time_limit`）——这是引擎空转到地平线的唯一情形，也是本小节标题的后半句。

没有引擎侧的早退：停车条件只依赖调用方给的数字，不依赖 `restart_rule` 的节奏，所以任何第三方规则都
不会让循环永不结束。空转的账在 `n_batches` 上是零，在 `n_iters` 上是满额——这就是为什么 `max_iters`
不是成本数字。

## 4. 适用界

- 这是**局部**方法 + 采样重启，不是全局搜索器；它靠 n_paths 条路径的多样性与重启频率换全局性，收敛
  保证与 Adam/RMSProp 同级，不追牛顿/拟牛顿。
- 逐坐标估计忽略耦合项 `Σ_{j≠i} Hᵢⱼ sⱼ`，后者落在 `|yᵢ|` 内使 `d` 成为 `1/Hᵢᵢ` 的有偏估计
  （耦合越强偏差越大，倾向系统性欠步：更稳但更慢）。这是拿「安全」换「潜在快」。
- 重启频率与 `n_draws`、`β`、`θ`、`τ_rel` 相关；文档只给结构，不给某类问题的推荐值——
  那些值由目标决定。

## 5. 两条成本轴（成本模型，不是评分）

一次求值服务一批 `(M, d)` 的点，所以同一次运行有两个代价数，它们回答不同的问题，**永远不相等**：

| 字段 | 数什么 | 回答的问题 | 用途 |
| --- | --- | --- | --- |
| `n_points` | 被计费的**点**（一个 g 批按其实际点数计点） | 串行效率：一个点买到多少改进 | 串行求值（逐点计费）的计分/门控口径 |
| `n_batches` | 被计费的**批**（一个 g 批计 1 批） | 并行效率：一次调用买到多少改进 | 批量目标（真 vmap/GPU）的代理口径 |

两个字段只数 **value+grad 融合批（g 批）**——“被计费”就是这个意思。f-only 的批（探测的抽、被拒的
trial）是便宜的那一半，既不进这两个数也不问闸（3.5）。**一次迭代花几个 g 批不是常数，由接缝决定**：

- 接了 `eval_f`：试用只花 f，真正欠的梯度（探测的起点 + 步的赢家）在轮末合成**一个** g 批，所以
  一次迭代是 0 个（全停滞的一轮，3.6）或 1 个。
- 不接 `eval_f`：批量求值是拿到值的唯一途径，于是**每个试用轮自己就是一个 g 批**——预条件方法按
  候选逐个 `retry`/`accept`，`n_alpha` 个候选就是一个判定步花掉 `n_alpha` 个 g 批，探测自带的那一批
  也在其中。所以 `n_alpha` 直接乘上 g 批数：实测（1 条路径、`n_alpha = 5`、跑 2 次迭代）不接接缝是
  6 个 g 批（1 个探测批 + 5 个试用轮），接缝下是 2 个（每个非停滞迭代一个合并批）。
  `n_draws` 不再往上乘——不接接缝时 `n_draws > 1` 直接被拒（3.2），接了接缝时多抽只是 f-only。

因此不接接缝时每个服务的批都是 g 批，`n_batches` 恰好等于服务批数；一旦接了 f-only 接缝，它就变成
“g 批数”（f 批与它分开数）——而后者才是承重轴（梯度次数）上的真账。

两个都要报：串行轴下，n_paths 条路径的收益会被 n_paths 除，多路径的账面上限被压低；并行轴下一次求值
就是一次并行调用的成本，「多样本 + 自适应重启」的收益才如实显形。两者都不等于墙钟（换底层语言能把
固定开销压掉，但改变不了「一次求值要服务几个点」这个结构量）。

停车记在迭代上（3.5），两个成本字段只报不预算：`n_batches` 与 `max_iters` 不同轴，所以「跑满
地平线用了多少批」必须读 `n_batches`，不能拿 `max_iters` 当成本数字去比。
**不许混写**：同一张表、同一张图只能用一条轴，且必须标注。把 `n_batches` 的数字
当成 `n_points` 的数字去比较，等于把 n_paths 倍的预算差抹平。

## 6. 可换的实现选择 vs 不许破的不变式

参考实现把接缝切成四块：局部方法（`restart/needs_verdict/propose/verdict/fold`）、路径点与 verdict 类型、
批量目标 callable、重启规则；另外允许注入起点抽样器。**这些签名全是实现选择**，新项目可以：

- 把局部方法写成纯函数 + 显式状态（不需要协议类）；
- 把规则写成闭包或普通函数（不需要一个 `_stage_window()` 方法）；
- 把批量目标写成任何可 vmap 的 callable，甚至写成一次同步阻塞的远程调用；
- 把 `PathPoint`/`Verdict` 换成元组/枚举/字符串——只要语义一致。

一条不是实现选择：枚举的长度归方法，所以 `verdict` 的终止责任也在方法（`accept` 或 `give_up`）。
引擎已经没有试用上限可退还，一个永远「再来一个」的方法就是挂死。

#### 必须保留的不变式（换了就坏）

1. `d ≥ 0` ⇒ `−d⊙g` 恒为下降方向 ⇒ 稳态不需要 acceptance test（1.2 之 1）。
2. 单位不变（1.2 之 2），以及旋钮全无量纲（1.2 之 3）。
3. 地板用 `max`，λ 只在 `Y < λ` 时生效（1.3）。
4. 两相共用一条 `alpha` 网格，`needs_verdict` 与相位无关；`n_alpha = 1` 时两相都是无护栏整步（1.4）。
5. 判决两侧的**样本数**相等：继续侧读最近 M 个、重启侧读全部 M 个（M = 池子长度）；空池时两侧都是
   0 个样本（两条读取都返回 `0.0`，而冷启动在读取之前就放行）。`window = M + 1` 只决定**何时**问，
   不是样本数；池子全路径共享（2.3）。
6. 单一 argmin 所有者：引擎决定谁赢，方法只通过 fold 得知（3.3）。
7. 预算门控在批边界，不切分批次（3.5）。

## 7. 测试带走清单

按「测什么」分三层。前两层随算法走，第三层是本仓边界、不带走。括号里是参考实现里的函数名——它们随这些
测试一起搬走，所以在新项目里同样可解析；换名字无所谓，语义在就行。

#### 算法核（跟着局部方法走，`test_solver.py`）

- 地板是 floor 而不是加数：`test_diag_floor_is_a_floor_not_an_addend`（构造 `Y = λ` 的交叉 secant，
  断言 `max` 形式、排除加法的 `1/√2` 因子）。
- 步机制只有一种：`test_diag_warmup_is_one_unruled_gradient_step`（`n_alpha=1` 的两相都不判定）、
  `test_diag_warmup_is_ruled_when_the_grid_compares_samples`（`n_alpha>=2` 的暖机照样走网格 argmin）。
- secant/幅值比：`test_diag_l2_step_is_newton_exact_on_a_quadratic`、
  `test_diag_abs_step_is_newton_exact_on_a_quadratic`（二次函数上 `d = 1/h`，一步到 0）。
- EMA 窗口：`test_diag_decay_blends_later_secants`。
- 地板只守 `Y→0`：`test_diag_gradient_floor_carries_a_flat_secant`（`y = 0`、`g ≠ 0` 仍有有限下降步）。
- 单位不变：`test_diag_step_scales_with_the_coordinate_unit`（同一 secant 在缩放坐标下 `d̃ = c²d`、
  `Δx̃ = c·Δx`；注意要注入 secant 对，不能走暖机——暖机步按设计就不是坐标不变的）。
- 停止条件：`test_diag_gives_up_only_when_the_gradient_norm_converged`、
  `test_diag_gives_up_on_a_nonfinite_trial`、`test_diag_judges_a_trial_on_its_value_alone`（判定只读服务来的值
  `f`——`x` 留在接缝里给需要几何的方法；非有限的梯度根本到不了方法）。
- 暖机网格丢非有限样本：`test_diag_warmup_grid_drops_a_nonfinite_sample`（非有限值只算 retry，全坏则 `give_up`）。

#### 引擎（跟着 n_paths 路径框架走，`test_solver_seam.py` / `test_scheduler.py` / `test_restart.py`）

- 接缝真的被驱动（而不是走默认）：`test_injected_solver_is_driven`、`test_injected_rule_is_driven`。
- f-only 求值器接缝：`test_f_only_probe_computes_one_gradient_per_probe`（`n_draws` 个只有值的抽
  + 一个赢家批，赢家仍是逐列 group-min；只有赢家批计费）、
  `test_a_single_draw_probe_reads_its_start_at_the_gradient`（单抽连 f 也不先读：没有可比的组，起点由
  补 g 的那一批一次求出 f 与 g，f 只算一次）、`test_a_rejected_trial_is_bought_no_gradient`
  （重试轮只花 f，整轮一步付一个 g 批）、`test_one_gradient_batch_per_iteration`（接缝下重启的起点与
  步的赢家同乘一个 g 批：形状是 `[(2,1),(2,1)]` 而不是两个宽 1 的批）、
  `test_multi_draw_probe_is_refused_without_the_f_only_seam`
  （不接 seam 时多抽必拒）、`test_the_gate_counts_g_batches_not_f_only_draws`（闸只数 g 批，f-only 抽
  照样发生却不进账）、`test_an_unjudged_step_never_buys_an_f_only_ruling`（不判定的步一次 f-only 读值
  也不买：值与梯度同在一批里只求一次，计费与判定步相同）。
- 起点要两半都有限：`test_a_non_finite_gradient_start_arrives_stalled`（`f` 有限、`g` 全 NaN 的
  group-min 不进线搜索，直接 stalled 等 EI）、`test_a_winner_with_no_usable_gradient_is_not_committed`
  （赢家只在提交点才被看见梯度：非有限则不 `fold`、按 stalled 报 `None`）、
  `test_an_unjudged_winner_must_be_finite_to_be_committed`（不判定的赢家连 `f` 也到提交点才被看见：
  两半任一非有限都不 `fold`）。
- 非有限事件表（一张表钉住所有局部裁决，`tests/test_nonfinite_matrix.py`）：把
  {有限, `+inf`, `-inf`, `nan`} × {可行, 不可行, 无 mask} 摊成 `_Best` 的 9 行（含「可行 `-inf` 让
  `fun_raw` 停在 `inf`」「不可行 `-inf` 什么都不改」两格），加上 `_argmin_finite` 的取值组合
  （`nan` 排前面不被选中、`-inf` 不算更小的有限值、并列取首个、全非有限返回 `None`）、`feas` 掩码的
  0/1 校验（`nan`、负裕度、`0.5` 一律 `ValueError`，且拒批时两格都不动）、以及「值好梯度坏」这一维
  的两条端到端（起步与赢家）。它不重覆上面各条机制测试，而是把它们的裁决合成一张表。
- 选择事件不盲降 + 批形状：`test_batch_shape_and_point_counting`、
  `test_probe_group_min_skips_nan`、`test_probe_inf_group_min_does_not_silence_restarts`。
- `stalled` 语义：`test_give_up_stalls_a_path`、
  `test_stalled_paths_wait_for_ei_then_restart`（到期就重探；窗口设到地平线之外的对照臂
  则一路空转到 `max_iters`）、
  `test_a_step_serves_as_many_trials_as_the_method_asks_for`（引擎不设试用上限：方法要 31 个 trial
  就给它 31 个）、`test_nonfinite_start_waits_for_ei_instead_of_stepping`。
- 单一 argmin 所有者：`test_diag_line_search_walks_the_grid_and_folds_the_argmin`、
  `test_diag_line_search_gives_up_when_no_sample_is_finite`。
- 判决节奏与可比样本数：`test_window_is_the_pool_plus_its_opening_round`、
  `test_the_two_sides_read_the_same_samples_the_same_way`、
  `test_the_continue_side_reads_only_the_newest_m_folds`、
  `test_verdict_is_due_once_per_window_and_closes_the_stage`。
- 停摆是缺席不是值、`+inf` 是死起步：`test_a_stall_supplies_no_delta_and_counts_the_iteration`、
  `test_a_stall_clears_a_partial_stage_and_keeps_its_wait`、`test_a_nonfinite_delta_is_rejected_at_the_seam`、
  `test_an_empty_pool_seeds_and_a_dead_pool_re_probes_once_per_window`。
- 池子共享 / stage 独立 / 锁相 / 重启保留池子：`test_group_min_pool_is_shared_across_paths`、
  `test_per_path_stage_is_independent`、`test_a_path_that_restarts_at_every_verdict_keeps_its_cohorts_rounds`
  （重启路径的探测轮也算一格，否则它每重启一次就落后同 cohort 一轮）、
  `test_restart_keeps_the_shared_group_min_pool`。
- 平局也重启（买第二个样本）、`n_paths = 1` 的退化与纯下降出口：
  `test_a_zero_payoff_tie_restarts_to_buy_a_second_sample` 覆盖「两侧同读 0 ⇒ 放行一次探测」,
  `test_no_best_still_restarts_at_a_due_gate` 与 `test_no_best_waits_for_the_gate_before_reprobing`
  覆盖 `f⋆` 缺失（到期即重探，不停在每轮），
  `test_a_dead_initialization_probe_does_not_freeze_the_run` 端到端地验证这一条：同一条死起步，
  换 `EIRestartRule` 能重新抽到有限点并收敛、换 `NoRestartRule` 则 `fun` 停在 `inf`，
  `test_no_restart_rule_seeds_once_then_never` 覆盖纯下降出口（`NoRestartRule` 只播种、之后永不再问），
  `test_restart_beats_pure_multistart_on_multimodal` 与 `test_k_start_beats_single_path_on_multimodal`
  则端到端地拿「同起点数、换规则」验证重启机制本身。
- 多抽 probe 只有单测覆盖：`test_probe_group_min_skips_nan`、
  `test_multi_draw_probe_is_refused_without_the_f_only_seam`、`test_the_gate_counts_g_batches_not_f_only_draws`。
  实验按 `n_draws=1`、不接 seam 跑（`f_only` 不计费，接入会白送评估），所以图里没有这条路径。
- EI 闭式：`test_normal_functions_match_known_values`、`test_ei_continue_degenerate_at_zero_bandwidth`、
  `test_ei_continue_is_a_mean_of_per_sample_ei`、`test_ei_restart_matches_kde_literal`、
  `test_ei_restart_is_a_mean_of_per_sample_ei`、`test_ei_restart_tail_below_all_observations`、
  `test_ei_restart_prices_every_nonfinite_start_as_a_zero_term`、`test_stage_read_is_a_kde_over_its_deltas`。
- **两条成本轴**：`test_reports_points_and_batches_separately`（`n_batches` 与 `n_paths` 无关；
  `n_points = n_batches × n_paths` 只在每个 g 批都是满 n_paths 批时成立，该测试正是这种全满批场景）。
- 停车在闸门上：`test_budget_exhaustion_stops_everything`、
  `test_a_run_without_an_extra_gate_reports_max_iters`、
  `test_the_gate_counts_g_batches_not_f_only_draws`（批级闸：重启风暴也超不过闸给的批数）、
  `test_every_path_feeds_once_per_iteration`（每迭代恰好一个样本，停摆的那个报 `None`）。
- 盒是数据：`test_box_is_data_not_a_unit_cube_assumption`、`test_bounds_spellings_are_the_same_box`、
  `test_unbounded_or_absent_box_needs_an_injected_sampler`、
  `test_the_sampler_is_the_only_authority_on_d`、`test_an_injected_sampler_searches_outside_the_old_box`。

#### 本仓边界（不带走）

`test_shell.py` / `test_shell_seam.py` / `test_submission.py`（dfbench shell 接缝、official-validator
冒烟）、数值指纹逐位比对、提交归档静态校验。这些依赖 dfbench 或交付包，与算法无关。

## 8. 与 SOAR 的关系

名字里的 "soar" 向 Mathesen et al. 2021 的**自适应重启**框架致敬，借的是它的**三分法**：全局侧（买
哪种起点划算）与局部方法各管一段，两者之间的重启/转移由第三个判定决定，而不是让局部搜索独自承担
全局多样性。后缀 `-lite` 指本包只做这三块的最小实现，且其中全局侧是**不拟合任何模型**的抽样池：重启
能买到多少，用起点组最小值**实测**（第 2.1 节），而不是先拟合一个 `f` 的代理模型再去问它。其余部分
是两回事：

- SOAR 是一个**全局–局部协同的优化框架**（文献含代理/集成模型与统计接受检验等成分）；本包不拟合
  任何 `f` 的模型、没有多保真，只有一个局部方法 + 一个重启判定。
- 重启判据不是 SOAR 的统计接受检验，而是共享 group-min pool 上 `EI_restart ≥ EI_continue` 的闭式比较
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
