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
（RMS），前者对小 secant 更平滑、后者对大 secant 更敏感。EMA 窗口 ≈ `1/(1−β)` 步，默认 `β = 0.63`
⇒ 约 `e` 步（2.7）。用幅值而非带符号的 `sᵢyᵢ` 是为了恒正：分母一旦能穿零或变负，`d ≥ 0` 就没有了。

**下降方向校正**：拿到 `d ≥ 0`，步取 `Δx = −diag(B)⁻¹ g = −d ⊙ g`。这一步的实质是把方向从
**梯度类量**改成**位移类量**——`−g` 的量纲是 `[f]/[x]`，直接当位移用，尺度会随坐标刻度反向走；
乘上 `d` 把 `1/H` 的量纲补进去。一次逐元素乘除完成，不需要矩阵、不需要解线性系统、不需要标量
学习率（第 1.4 节会用到「位移类」这个性质）。

### 1.2 两条不变式 + 一个 `μ = 0` 时的方向性质

1. **下降方向（仅在 `μ = 0`）**：`d ≥ 0` 由构造保证（绝对值/平方/开方 + 正地板），所以
   `(−d⊙g)·(−g) ≥ 0` 逐坐标成立。预条件后仍是下降方向 ⇒ **`−d⊙g` 的步不需要任何 acceptance test**，
   不需要线搜索去「验证方向」。`μ > 0` 时方向换成当前动量 `m_k`（第 1.6 节），这一条**不再成立**——
   第 1.6 节说明为什么放弃下降方向、收敛改由引擎层承担。默认 `μ = 0.63`，所以**默认不持有这一条**；
   `μ = 0` 时逐位成立。
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
- **单位重标定残差**：`experiments/RESULTS.md` 量到的 1.6× 点数差**整个**由暖机回溯次数造成
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

### 1.6 动量混合：当前动量替代当前梯度

`μ > 0` 时方向不再只用本步梯度，而是用**当前动量**——刚被 `g_k` 更新过的梯度 EMA：

```text
m_k = (1−μ)·g_k + μ·m_{k−1},        μ = 0 时 m_k = g_k
方向 = m_k
```

`m` 是历史起点梯度的 EMA（窗口 ≈ `1/(1−μ)` 步，读写时机见下）。方向 `m_k` 就是**不投影的
heavy-ball 方向**（`m_k = g_k + μ(m_{k−1} − g_k)`，两项展开是 `g` 的指数衰减组合），也是 Adam 的
first moment（`m_t = β₁m_{t−1} + (1−β₁)g_t`）在纳入其分母之前的那部分。单位不变照旧：`m` 与 `g`
同量纲，`m_k` 仍是**梯度类量**，乘上 `d` 之后仍是位移类（1.2 之 2、之 3）。

**不投影，也不保下降方向。** 早期版本把 `m` 投影到 `g` 的 `d`-度量正交补（方向 `g + m⊥`），换来
`⟨d⊙(g+m⊥), g⟩ > 0` 的下降方向不变式。两个理由把它否掉：

1. **投影砍掉的是动量唯一有用的那半。** heavy-ball 的价值在沿 `g` 的分量 `m∥`——压 zigzag、病态二次
   的震荡阻尼、平坦区的动量积蓄；横向分量 `m⊥` 只做「把方向掰转」，而「非对角耦合」真正想要的正是
   `m∥` 的过冲抵消。「投影保下降」等于为了一个不变式，把动量在它唯一能帮忙的地貌上的作用删掉。
2. **下降方向对一个没有线搜索的方法兑现不了下降。** 不变式只保证「存在充分小的步使 `f` 降」，而本
   方法 `n_alpha = 1` 时一次无护栏整步、不读值不回溯不与起点比较，`n_alpha ≥ 2` 的网格 argmin 也不
   判「比起点好」——下降方向并不自动生成下降。要买「这一步真的降」得装 Armijo / 线搜索，本项目
   明确不装：收敛本就不靠每步下降，靠引擎层的 best-so-far 记账 + restart + stall。既然如此，投影
   辛苦维持的「下降方向」就只剩一个空壳，随之删除；需要该性质的调用方自己减小 `α` 或在接缝外做
   线搜索，不在方法内硬保。

**「加速」不引 Nesterov。** Nesterov 的 O(1/k²)（凸光滑界；二次上单步复原牛顿）来自把梯度差
`y = g_k − g_{k−1}` 当二阶信息；本方法的 `d = S/Y` 已经是同一份二阶利用，且**量纲闭合**——`y` 除以
同向位移 `s` 才得到位移类的 `H⁻¹` 估计。Nesterov 只吃 `y`、不除 `s`，二阶量纲被藏进它的步长 `1/L`
（`L` 是曲率，属 `[f]/[x]` 系），相比本方法少了合理性；且它的前瞻点要查 `g`，在 propose→oracle 的
接缝下塞不进去。故不引入，让 `μ` 只做「方向噪声的平滑」这一件事。

**默认 `μ = 0.63`：与 `β` 同一个窗口，约 `e` 步。** 动量的代价是真的：近最优点 `g → 0`，
而 `m` 按窗口 `1/(1−μ)` 滞后衰减，`m_k` 里的历史项 `μ·m_{k−1}` 可以比 `g_k` 还大，方向被旧梯度
带走，把一步到位的精确步刮成绕路，同样的步数下降得更少。甜点上直接可观
（`experiments/RESULTS.md`）：对角二次形实测从 `1.3e-60`（`μ=0`）退到 `6.6e-9`（`μ=0.9`）。动量只在
`d` **结构性错**的地方（旋转 Hessian 的非对角曲率，逐坐标比值表达不了）换到进度，且只是「把旧方向
再用一次」的缓解，不是把缺的曲率补上——不是修复。默认取 0.63 而不是 0，是把「窗口」当这两个平滑
共有的唯一自由度来选：没有理由给方向猜一个比曲率长一个数量级的窗口。收益与代价都由重跑去读，不在
文档里先验断言；`μ = 0` 仍是关掉它的开关，逐位回到纯梯度方向。

**`μ` 与 `β` 是两个独立旋钮，只是默认窗口相同（`β = μ = 0.63`）。** `β` 平滑的是**曲率**估计
（`s`/`y`），回答「`d` 多久认清一次几何」；`μ` 平滑的是**方向**噪声。两者都以「几步之内几何/方向
变一次」为同一个物理尺度，所以取同一个 `W ≈ e` 作默认；但两者仍是两个旋钮，改一个不动另一个。
`s` 与 `y` 仍共用一个 `β`：它们是
同一个割线对的两端，`d = S/Y` 是比值的估计，两端窗口不同会在变局时把比值本身弄偏（所以也不给 `s`、
`y` 各配一个 `β`）。耦合点只有一处：`m` 只喂方向，割线对仍读**原始** `g` 差（`y = g_k − g_{k−1}`，
不经 `m`），否则两个旋钮就纠缠了。折点唯一：`fold`（步 commit 时）写 `m`，`propose` 只读，所以
`n_alpha > 1` 的网格对同一步重采多次也只读同一份历史，不存在重复折叠缩短 `μ` 买到的窗口；
`restart` 不碰 `m`，它只把 `has_secant` 置回 False，下一次 `fold`（暖机那次）用新起点的梯度重种——
和 `s`/`y` 的重种走的是同一条路径，历史因此不跨重启。

## 2. 重启：停滞触发，落点交给模型

### 2.1 默认不拟合代理模型

一次重启要做的只有一件事：给一条已经停下的路径换一个新起点。这个起点要么从 `Domain` 直接抽
（`SampleGlobalModel`，默认），要么让一个在已服务样本上拟合出的代理模型去指（`BLRGlobalModel`，2.3）。
默认走前者：抽一个点只要一次求值、不拟合任何东西，成本与维数无关。局部方法的甜点是「`d` 对的地貌」，
在那里重启本来就该少；为它先备一个代理，是拿全局侧的成本换大多数运行用不上的拐杖。

### 2.2 三个问题分开：何时、是否、何处

重启由三个互不相同的决定拼成，本实现把它们放在三个不同的地方，谁也不替谁回答：

- **何时**：路径停下是局部方法自己的判断（1.5 的 `τ_rel` 停止测试，或一次没有任何有限样本可赢的判定）。
  引擎不设第二个停止规则，也不猜「还差多少」。
- **是否**：`restart_stalled`，一个布尔旋钮，默认 `True`。关着时停下的路径就一直停着——`stalled` 仍
  记账、仍占位、它的点仍可读——只是不再被移动；开着时它下一轮立刻被重画。
- **何处**：`global_model.propose(n, domain)`，引擎的第二个插件点（2.3）。

这个切法是有意的：*何时* 已经由方法的停止测试说清，*是否* 是一个策略位，只有 *何处* 需要真正的实现，
所以只有 *何处* 成了一道接缝。上一版把三者揉进一个「重启判定」里（比较继续还是重来的期望改进，再按
判决重画），代价是一块必须自己攒证据、自己推导节奏的第三机制；现在那层没有存在的必要——停下的路径
要么原地不动，要么直接换点，中间没有要比较的量。

### 2.3 两个落点臂

**抽样臂（默认）。** `SampleGlobalModel` 不拟合，`observe` 是空操作，`propose` 只是把 `Domain.draw`
要来的点原样交回：有界侧均匀、自由侧 `N(0, scale²)`、半界侧同正态截断、调用方给了逐维抽样器就按注入
的来。羊群不归它管：第一轮的起点由入口的 `x0` 直接给出，它只在路径真的重启时才被问到。
所以一次重启的边际成本是一个点、一次求值，与维数无关。

**代理臂。** `BLRGlobalModel` 在已服务样本上拟合一个 Bayesian 线性后验（特征由注入的 `Projection`
给出），`propose` 返回 `MinimizeFn` 在盒内对 `−log EI` 求得的局部极小。**EI 在这里的身份是采集函数**
（Jones et al. 1998），用来回答「代理认为哪里还值得试」，而不是「继续还是重来」的重启判决——它交出
一个起点，不做任何比较。`−log EI` 的下降用调用方注入的 `MinimizeFn`（通常就是本包自己的 n_paths 引擎），
所以这一臂是「用一次内层搜索换一个更聪明的起点」，代价是内层那轮求值。

### 2.4 重启不额外花一次梯度

`propose` 的答案被 clip 进 `domain`、包成**两半都欠**的点，和本轮各路径步的赢家一起交给 `settle`，
所以重启起点不会单独买一个梯度批。接上 f-only 接缝时这份合并更彻底：重启起点的值也留到同一个融合批
里一次求。一个不可用的起点（`f` 或 `g` 非有限）不落地、路径留在 `stalled`，下一轮再画——它不会变成
`points[i]`，所以「这条路径站在哪」不会被一个退化点污染。

**要一条一步都不重启的纯多起点下降，把 `restart_stalled` 显式关掉**：停下的路径原地不动，
不再花预算，也不会被任何量唤醒；羊群全停时运行就以 `all_stalled` 结束（3.6），而不是空转到地平线。

## 3. n_paths 路径引擎

### 3.1 一次迭代的日程

一轮先分路径，再一批买：

1. **重画集**：没有点（只有 iteration 0 的占位符）或「`stalled` 且 `restart_stalled`」的路径。
2. **队列**：其余路径。已 `stalled` 但运行不重画的路径留在这里——它这一轮不提案、不产生赢家，但它的
   点仍然可读（`f⋆` 从它那里读得到）。
3. **落点**：重画集向 `global_model.propose(len(restarted), domain)` 要一批起点。引擎校验形状、clip 进盒；
   形状不对直接 `ValueError`——宽度归引擎所有，模型返回别的宽度在批次边界就失败，而不是深埋在广播里。
4. **步**：队列里的每条路径按 `needs_verdict` 分两条通道（3.2），产出各自这一步的赢家。
5. **一批结清**：重画起点在前、步赢家按路径序在后，**同一个**批次交给 `settle` 补上仍未买的梯度。
6. **落地**：起点走 `_place`（不可用则留在 `stalled`、下轮再画），赢家走 `_advance`（不可用则 `stalled`、
   不 `fold`）。

重画集与队列不相交，所以「先定完再一起买」与「逐条定完就买」是同一日程。

### 3.2 判定通道与不判定通道

`needs_verdict(i, p)` 问的是「这条路径下一步要不要逐候选判定」。默认 `n_alpha = 1` 的整步答 False：
方法提交即所提，引擎把它的一次提议当作这一步的赢家、**不先读值**，值与梯度留到 `settle` 的融合批里
一次求。`n_alpha > 1` 的步长网格答 True：方法逐候选 `propose`，引擎 clip 后服务，方法 `verdict` 读
`retry`/`accept`/`give_up`。

判定通道里的失败只有两条出路，两条都由方法给出：`give_up`（它的收敛测试触发）置 `stalled`；`accept`
但没有服务过任何有限样本也置 `stalled`。引擎侧没有试用上限——枚举多长是方法自己的事，加一个上限
就是给一个已有决定再加第二个隐形停止规则；代价是一项契约：`verdict` 必须最终返回 `accept` 或
`give_up`，永不终止的枚举是挂死而不是慢跑。

接 f-only 接缝时，判定通道的每个 trial 只服务 f（判决只听值），赢家的梯度留到 `settle`；不接接缝时
每一轮 trial 自己就是融合批，所以 `n_alpha` 个候选直接乘上融合批数（5 节）。

一个起点要能在优化里用，`f` 和 `g` **两半都得有限**才算数：`f` 有限但梯度非有限的起点会被
`propose` 读取（`-p.g`）并进 `fold`（`best.g - p.g`），一个 NaN 就能把那个坐标的量级 EMA 永久毒死。
这样的起点按「死起步」处理、到达即 `stalled`，由 `restart_stalled` 决定要不要重画（3.4）。

### 3.3 单一 argmin 所有者与 `fold`

一次判定步里引擎提交**所有被服务的 trial 中 `f` 最低的那个**，不是最后一个。**引擎是「谁赢」的唯一
所有者**：方法只通过 `fold(i, step_start, winner)` 得知落点，不自己维护第二份 argmin，重启与折叠的
簿记因此不会与方法的内部状态漂移。

方法的判定只收**坐标与值**：`verdict(i, p, x, f)`，`p` 是这一步的起点，`x` 是 clip 后实际服务的那一点。
trial 的梯度刻意不在契约里——引擎只为赢家买 g，方法拿不到别的 trial 的梯度，需要它才能判决的方法
不是这个引擎能驱动的对象。相应地，赢家的梯度在**提交点才第一次被看见**（`f` 先定，g 随收），不判定的
步连 `f` 也是这时才第一次看见（3.2）。所以提交闸对**两半**都查：`f` 或 `g` 任一非有限就不提交、按
`stalled` 处理。这样 `fold` 是唯一的提交点，方法不会先被告知「落点」再被收回。

### 3.4 `stalled` 是唯一的路径状态

一条路径在一轮里既没被重画、也没提交步，就叫 `stalled`。它的写者只有两个：方法自己的 `give_up`
（或「接受但没有有限样本」），以及提交闸对非有限起点/赢家的拒绝。一条被拒绝的起点**不会**成为
`points[i]`——`points[i]` 永远表示「这条路径站在哪」，所以不可用的起点留在 `stalled`、下一轮重抽，
而不是被 clip 或强塞成一个退化点。

`stalled` 与「没有点」是两件事，且只有后者不由 `restart_stalled` 管：占位符的空 `x` 是机器自己的初始
状态（iteration 0 借此让所有路径都走一次重画），所以它总是重画；握着一个点但值非有限才是停摆，由
`restart_stalled` 决定要不要重画。

### 3.5 停车是三闸门，记账只报告

`max_iters` 是引擎自己的地平线，单位是**迭代**（iteration 0 是放置羊群的那一轮）。它是确定性的停车
界，不是成本代理——一轮服务几个批由方法和接缝决定，从重画起点的那一批到方法的若干试用批。
`time_limit` 是同一件事的墙钟版本，给「一轮多贵说不准」的调用方。两者至少给一个，否则入口抛
`ValueError`。`budget_check` 是给引擎看不见的预算（点数、墙钟、钱、外部早停）的额外闸。

`_Budget` 因此合了两个职责：停车计量与工作量报告。**两对计数都记账**（融合对与 f-only 对，5 节），但
**计数从不决定停车**：`stop` 只读三个谓词（地平线、墙钟、调用方的闸）。`charge` 只记账，`tick` 在
一轮完成后记一次迭代，`stop` 是唯一的停车查询——三个输入报一个拒绝的理由，拒绝不存成状态，否则就
和自己的证据脱钩。

同一个 `stop` 也是循环头的停车条件：一次完成的迭代可能刚好把地平线走到，所以在开新一轮之前先查一次。
迭代中途被拒则在请求点抛 `_BudgetExhausted`（理由装在这条异常里）并结束，该轮不计入 `n_iters`：
中间层因此不必检查返回值、也不必上下传任何标志。**每个批之前都问一次闸，融合批与 f-only 批都一样**
——判定通道的 `while` 可以转很多圈而不回到循环头，没有这一问，一次跨过截止时刻的步骤会一路跑完。

`status` 按 `max_iters` → `time_limit` → `budget_exhausted` 的顺序报第一个成立的闸。另外两种停车是引擎
自己认出的：

- `unbounded`：某个 **feasible** 点把目标打到 `-inf`，那里没有下界，任何后续试用都比不过它、任何一步也
  改不动它，所以该点直接成为答案（`fun = -inf`，`x` 就是它）并结束运行。它仍然读可行性掩码：不可行的
  `-inf` 什么都不改。找到它的那一批已服务并记账，但这一轮没跑完，因此和批被拒一样不入 `n_iters`。
- `all_stalled`：关着重启时羊群自锁，没有任何路径能动、也没有任何路径欠起点（3.6）。同样不入 `n_iters`。

四个计数只报不预算：引擎无法在一个不可分的批内部设卡。所以 `budget_check` 那类外部预算可能超跑不到
一个批大小——一个有意的代价，靠「g 约为 f 的 2–5 倍、f 便宜」的前提兜底；f 本身贵的调用方就不该接
f-only 接缝。

### 3.6 全停：重画，或者结束运行

所有路径都 `stalled` 时，这一轮既没有任何步、也没有任何欠账的起点，只剩两种结局，由
`restart_stalled` 一个位决定：

- **开着（默认）**：停摆的路径下一轮就进重画集，于是每一轮都有批可服务——羊群在停滞与重画之间循环，
  间隔由局部方法停下的快慢决定，不由引擎的窗口决定。
- **关着**：停下的路径不会自己复活，羊群自锁；引擎认出这个死锁并结束运行，`status` 报 `all_stalled`。
  这一轮一个批都没服务，所以和批被拒、`unbounded` 一样不入 `n_iters`。

死锁的判据是「重画集为空且每条路径都 `stalled`」：重画集空 ⟺ 每条路径都握着点且没人要重画，`all(stalled)`
⟺ 每条路径都停着。开着时前一半永远不成立（停摆的路径总在重画集里），所以 `all_stalled` 只在关着时可及，
而它正是关着时唯一能打破自锁的东西。

没有别的引擎侧早退：`all_stalled` 之外的停车条件只依赖调用方给的数字，所以任何插件都不会让循环永不结束。

## 4. 适用界

- 这是**局部**方法 + 采样重启，不是全局搜索器；它靠 n_paths 条路径的多样性与重启频率换全局性，收敛
  保证与 Adam/RMSProp 同级，不追牛顿/拟牛顿。
- 逐坐标估计忽略耦合项 `Σ_{j≠i} Hᵢⱼ sⱼ`，后者落在 `|yᵢ|` 内使 `d` 成为 `1/Hᵢᵢ` 的有偏估计
  （耦合越强偏差越大，倾向系统性欠步：更稳但更慢）。这是拿「安全」换「潜在快」。
- 重启频率由 `restart_stalled` 与局部方法的停止测试（`β`、`μ`、`θ`、`τ_rel`）共同决定；文档只给结构，
  不给某类问题的推荐值——那些值由目标决定。

## 5. 成本轴：两道接缝 × 两把尺子（成本模型，不是评分）

一次求值服务一批 `(M, d)` 的点，所以同一次运行有两把尺子，它们回答不同的问题、**永远不相等**：

| 尺子 | 数什么 | 回答的问题 |
| --- | --- | --- |
| `points` | 被计费的**点**（一个批按其实际点数计点） | 串行效率：一个点买到多少改进 |
| `batches` | 被计费的**批**（一个批计 1） | 并行效率：一次调用买到多少改进 |

引擎有两道求值接缝（`eval_f_g` 融合批、`eval_f` 只有值），各有一对计数，所以结果是四个字段：

| 字段 | 数什么 |
| --- | --- |
| `n_f_g_points` / `n_f_g_batches` | 融合批的点数 / 批数 |
| `n_f_points` / `n_f_batches` | f-only 批的点数 / 批数 |

**两对都记账**——「被计费」就是「真的服务出去了」的意思。旧的「只有 g 批计费、f-only 白送」口径已废：
调用方可能自带一个只给值的 `eval_f`，那半边的工作量必须如实报出来，否则串行轴在接了接缝的运行里会把
f 的成本整个抹掉。计数只报不预算：`stop` 从不读它们（3.5）。

**一轮花几个批由接缝决定**：

- **接了 `eval_f`**：trial 与重画起点都先只服务 f，欠的梯度在轮末合成**一个**融合批，所以一轮是 1 个
  融合批，外加方法判定所需的 f-only 批。
- **不接 `eval_f`**：融合批是拿到值的唯一途径，于是**每个 trial 自己就是一个融合批**——预条件方法按
  候选逐个 `retry`/`accept`，`n_alpha` 个候选就是这一步花掉 `n_alpha` 个融合批，重画起点自带的那一批
  也在其中。实测（1 条路径、`n_alpha = 5`、跑 2 次迭代）：不接接缝 6 个融合批，接缝下 2 个。

两把尺子都要报：串行轴下 n_paths 条路径的收益会被 n_paths 除，多路径的账面上限被压低；并行轴下一次
求值就是一次并行调用的成本，「多样本 + 重启」的收益才如实显形。两者都不等于墙钟（换底层语言能把固定
开销压掉，但改变不了「一次求值要服务几个点」这个结构量）。

停车记在迭代上（3.5），四个字段只报不预算：`batches` 与 `max_iters` 不同轴，所以「跑满地平线用了多少
批」必须读 `n_*_batches`，不能拿 `max_iters` 当成本数字去比。**不许混写**：同一张表、同一张图只能用一条
轴、且必须标注，也不能把融合轴的数当成 f-only 轴的数去比。

## 6. 可换的实现选择 vs 不许破的不变式

参考实现把接缝切成四块：局部方法（`restart/needs_verdict/propose/verdict/fold`）、全局模型
（`observe/propose`）、路径点与 verdict 类型、批量目标 callable；另外 `Domain` 允许注入逐维抽样器。
**这些签名全是实现选择**，新项目可以：

- 把局部方法写成纯函数 + 显式状态（不需要协议类）；
- 把全局模型写成闭包或普通函数（不需要一个攒样本的 `observe`）；
- 把批量目标写成任何可 vmap 的 callable，甚至写成一次同步阻塞的远程调用；
- 把 `PathPoint`/`Verdict` 换成元组/枚举/字符串——只要语义一致。

一条不是实现选择：枚举的长度归方法，所以 `verdict` 的终止责任也在方法（`accept` 或 `give_up`）。
引擎已经没有试用上限可退还，一个永远「再来一个」的方法就是挂死。

#### 必须保留的不变式（换了就坏）

1. `d ≥ 0` 由构造保证 ⇒ `−d⊙g` 在 `μ = 0` 时恒为下降方向 ⇒ 稳态步不需要 acceptance test
   （1.2 之 1）。`μ > 0` 时方向换成当前动量，这一条不再成立（1.6）。
2. 单位不变（1.2 之 2），以及旋钮全无量纲（1.2 之 3）。
3. 地板用 `max`，λ 只在 `Y < λ` 时生效（1.3）。
4. 两相共用一条 `alpha` 网格，`needs_verdict` 与相位无关；`n_alpha = 1` 时两相都是无护栏整步（1.4）。
5. 单一 argmin 所有者：引擎决定谁赢，方法只通过 `fold` 得知（3.3）。
6. 预算门控在批边界，不切分批次；每个批之前都问一次闸，不过滤也不合并（3.5）。

## 7. 测试带走清单

按「测什么」分三层。前两层随算法走，第三层是本仓边界、不带走。括号里是当前实现里的函数名——它们随这些
测试一起搬走，所以在新项目里同样可解析；换名字无所谓，语义在就行。

#### 算法核（跟着局部方法走，`test_local_solver.py`）

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
- 动量（1.6）：`test_diag_momentum_keeps_the_plain_step_descent_gain`（`μ > 0` 仍满足 `⟨g,g⟩_d > 0`）、
  `test_diag_momentum_folds_only_at_the_commit_not_in_propose`（历史只在 `fold` 写入，`propose` 是纯读）、
  `test_diag_mu_zero_keeps_no_momentum_state`（默认路径不维护任何动量状态）、
  `test_diag_momentum_is_skipped_when_g_has_no_d_support`、`test_diag_restart_drops_the_momentum_history`。
- 重启复位：`test_diag_restart_rearms_the_warmup_and_reseeds_g0`（`g0` 重新播种、暖机重新武装）。
- 旋钮校验：`test_diag_knobs_out_of_range_are_rejected`（`β`、`μ`、`n_alpha`、`backtrack` 的边界）。
- 端到端：`test_diag_drives_a_quadratic_to_its_minimum_through_the_engine`、
  `test_diag_momentum_runs_through_the_engine`。

#### 引擎（跟着 n_paths 路径框架走，`test_scheduler.py` / `test_plugin_seam.py`）

- 接缝真的被驱动（而不是走默认）：`test_injected_solver_is_driven`、
  `test_the_default_model_is_exactly_a_sample_global_model`。
- 停摆与重画：`test_give_up_stalls_a_path`（`give_up` 置 `stalled`）、
  `test_a_dead_start_is_redrawn_without_asking_the_rule`（不可用的起点不落地，下一轮重画）、
  `test_a_stall_hands_a_redraw_to_the_global_model`（开着 `restart_stalled` 时停摆的路径向模型要起点）、
  `test_a_frozen_flock_that_fully_stalls_ends_the_run`（关着时全停是 `all_stalled`，不是空转到地平线）。
- 起点与赢家两半有限：`test_an_unusable_start_is_refused_before_it_seeds_the_method`、
  `test_a_non_finite_gradient_start_arrives_stalled`（`f` 有限、`g` 全 NaN 的起点到达即 `stalled`）、
  `test_a_winner_with_no_usable_gradient_is_not_committed`（非有限则不 `fold`）,
  `test_an_unjudged_winner_must_be_finite_to_be_committed`（不判定的赢家连 `f` 也到提交点才被看见）。
- 一次迭代一个融合批：`test_one_gradient_batch_per_iteration`（重画起点与步的赢家同乘一个批：形状是
  `[(2,1),(2,1)]` 而不是两个宽 1 的批）、`test_a_flocks_start_is_served_once_fused`、
  `test_a_rejected_trial_is_bought_no_gradient`、`test_an_unjudged_step_never_buys_an_f_only_ruling`。
- 四字段记账：`test_reports_four_counts_on_two_axes`、`test_batch_shape_and_point_counting`。
- 闸：`test_budget_exhaustion_stops_everything`、`test_a_run_without_an_extra_gate_reports_max_iters`、
  `test_the_gate_is_asked_before_a_value_only_batch_too`、
  `test_the_gate_counts_g_batches_not_f_only_draws`（调用方自记的账单仍以融合批为准；f-only 照样被数、
  被报，只是不闭合闸）。
- 不设试用上限：`test_a_step_serves_as_many_trials_as_the_method_asks_for`（方法要 31 个 trial 就给它 31 个）。
- 非有限事件表（一张表钉住所有局部裁决，`tests/test_nonfinite_matrix.py`）：`_Best` 的可行性 × 有限性
  （`test_best_counts_a_value_only_when_it_is_feasible_and_finite`）、`_argmin_finite` 的取值组合
  （`test_argmin_finite_never_returns_a_non_finite_candidate`）、`feas` 掩码的 0/1 校验
  （`test_a_mask_flag_that_is_not_zero_or_one_is_refused`、`test_a_mask_of_zero_and_one_flags_is_accepted`），
  以及「值好梯度坏」这一维的两条端到端（`test_a_half_dead_start_stalls_the_path_but_its_value_still_reports`
  与 `test_a_half_dead_winner_is_refused_before_it_becomes_a_start`）。
- 盒是数据、不是单位立方体假设：`test_box_is_data_not_a_unit_cube_assumption`、
  `test_bounds_spellings_are_the_same_box`、`test_the_domain_is_the_runs_only_authority_on_d`、
  `test_an_unbounded_box_is_really_unbounded`、`test_a_model_that_oversteps_is_clipped_into_the_runs_box`。
- `Domain` 抽样（`test_domain.py`）：`test_the_default_scale_is_one_not_the_box_width`、
  `test_a_mixed_box_draws_the_unbounded_dimension_at_its_own_scale`、
  `test_a_half_bounded_draw_is_the_half_normal_not_the_full_one`、
  `test_an_injected_sampler_replaces_that_dimensions_default`、
  `test_a_sampler_that_returns_the_wrong_shape_is_refused`。
- 模型契约（`test_global_model.py`）：`test_a_model_free_domain_draws_its_unbounded_dimension`、
  `test_the_draws_are_the_handed_domains_own`——抽样臂只证明它照盒抽样，第一轮的羊群不在它的契约里。
- 羊群（`test_scheduler.py`）：`test_the_caller_places_the_flock_and_the_model_is_not_asked_for_it`、
  `test_without_x0_the_model_places_the_flock`、`test_x0_is_placed_once_and_every_later_restart_is_a_draw`、
  `test_a_start_on_the_wall_is_inside_the_box`、`test_a_start_outside_the_box_is_refused_not_clipped`、
  `test_only_roundoff_past_a_wall_is_tolerated`、`test_the_box_test_reads_only_the_declared_sides`。
- 代理臂（`test_spherical_linear.py`）：`test_propose_draws_uniformly_until_a_fit_is_possible`、
  `test_propose_descends_the_acquisition_into_the_box`、`test_posterior_variance_stays_under_the_prior_far_from_data`、
  `test_log_h_matches_its_definition_where_it_can_be_written_down`、`test_the_model_carries_no_box_of_its_own`。
- 入口校验：`test_out_of_range_knobs_are_rejected_at_the_entry`、`test_a_nonpositive_width_is_refused_at_the_entry`、
  `test_a_flock_of_the_wrong_shape_is_refused_at_the_entry`、`test_a_malformed_flock_is_refused_at_the_entry`、
  `test_a_seam_dial_that_is_not_callable_is_refused_at_the_entry`、
  `test_a_seam_object_without_its_methods_is_a_type_error`。

#### 本仓边界（不带走）

dfbench 的 shell 接缝与 official-validator 冒烟、数值指纹逐位比对、提交归档静态校验。它们依赖 dfbench 或
交付包，与算法无关，所以不在本包里。

## 8. 与 SOAR 的关系

名字里的 "soar" 向 Mathesen et al. 2021 的**自适应重启**框架致敬：它主张全局侧（买哪种起点划算）与
局部方法各管一段，两者之间的重启/转移由第三个机制决定，而不是让局部搜索独自承担全局多样性。`-lite`
指本包只做这套骸架的最小实现：重启起点由 `GlobalModel` 给，默认那一臂不拟合任何模型，直接按 `Domain`
抽样（2.1、2.3）。其余部分是两回事：

- SOAR 是一个**全局–局部协同的优化框架**（文献含代理/集成模型与统计接受检验等成分）；本包默认不拟合
  任何 `f` 的模型、没有多保真，只有一个局部方法 + 一个起点模型接缝。
- SOAR 的第三机制判「继续还是重来」并自带证据积累；本包把它拆成 `restart_stalled` 一个位（是否）与
  `GlobalModel.propose`（何处），停下的判定留给局部方法自己的停止测试（2.2）。代理臂的 `−log EI` 是
  采集函数，服务于「哪里值得再试」，不是重启判决（2.3）。
- 局部方法默认是逐坐标 secant 幅值比预条件（第 1 节），一阶，无 Hessian、无曲率模型。

本包与 SOAR 的算法**核心一致**——全局侧与局部方法各管一段、由一个衔接机制串起来；**细节不同**——
`-lite` 是这套骨架的最小实现，不是照 SOAR 文献的实现搬来的。所以它**不是** SOAR 的纯粹移植，也不
暗示接口兼容或结果可复现。引用 SOAR 仅为归属与差异定位（第 2.1 节正文）。

## 9. 参考文献

- Conn, A. R., Gould, N. I. M., & Toint, P. L. (2000). *Trust-Region Methods*. SIAM/MPS.
  （割线方程 `B s = y` 与 SR1 更新；后者的对角退化就是本文的比值）
- Mathesen, L., Pedrielli, G., Ng, S. H., & Zabinsky, Z. B. (2021). Stochastic optimization with
  adaptive restart: a framework for integrated local and global learning. *Journal of Global
  Optimization*, 79, 87–110.（SOAR 式样本重启）
- Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive
  black-box functions. *Journal of Global Optimization*, 13, 455–492.（EI 闭式）
