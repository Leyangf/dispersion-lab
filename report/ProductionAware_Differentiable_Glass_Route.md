# 面向生产约束的连续玻璃材料优化路线

## 1. 核心判断

当前项目最有价值的主线不是“用神经网络替代 Sellmeier”，而是：

$$
(n_d, V_d, \Delta P_{g,F})
\rightarrow
n(\lambda)
\rightarrow
\text{differentiable optical design}
$$

也就是把离散玻璃目录连续化，使材料选择能够进入可微光学设计流程。

在这个框架中：

- Notebook 01 提供 anchor-preserving Buchdahl 材料层；
- Notebook 02 验证该材料层对可见光 achromat / apochromat 玻璃替换有设计意义；
- Notebook 03 提供可微光学设计框架；
- 材料可购性、热特性、透过率、成本、加工限制应优先接入 Notebook 03；
- Notebook 05 的 neural residual 不应替代 Notebook 03，而应作为未来可插入 Notebook 03 的更高保真材料色散层。

因此，长期架构应是：

```text
Differentiable optical design framework        <- Notebook 03
    |
    |-- Material layer A: Buchdahl K4           <- Notebook 01
    |
    |-- Material layer B: Buchdahl + NN residual <- Notebook 05, future broadband extension
    |
    |-- Catalog / thermal / cost / manufacturing constraints
```

## 2. 为什么生产约束应该优先和 Notebook 03 结合

Notebook 03 的本质是系统级优化框架：

$$
x=(n_d,V_d,\Delta P_{g,F})
\rightarrow
n(\lambda)
\rightarrow
\text{ray tracing / merit function}
\rightarrow
\text{gradient optimization}.
$$

如果要考虑真实生产和加工，最自然的做法是在 Notebook 03 的 loss 中加入材料和制造约束：

$$
L =
L_{\text{optical}}
+
\lambda_{\text{cat}}L_{\text{catalog}}
+
\lambda_{\text{thermal}}L_{\text{thermal}}
+
\lambda_{\text{trans}}L_{\text{trans}}
+
\lambda_{\text{cost}}L_{\text{cost}}
+
\lambda_{\text{mfg}}L_{\text{mfg}}.
$$

其中：

- \(L_{\text{optical}}\)：光学性能目标，例如焦移、色差、spot、MTF 或 wavefront；
- \(L_{\text{catalog}}\)：连续材料点到真实玻璃目录的距离；
- \(L_{\text{thermal}}\)：温度漂移、\(dn/dT\)、athermal behavior；
- \(L_{\text{trans}}\)：透过率或吸收约束；
- \(L_{\text{cost}}\)：成本等级或供应风险；
- \(L_{\text{mfg}}\)：几何加工限制和材料加工限制。

这比把制造性强行塞进 Buchdahl 公式或 neural residual 更合理，因为加工性本质上是系统级设计约束，而不是单纯的 \(n(\lambda)\) 曲线拟合问题。

## 3. 材料可购性：最应该优先实现

真实玻璃目录可以看成三维点云：

$$
\{x_i\}
=\{(n_{d,i},V_{d,i},\Delta P_{g,F,i})\}.
$$

连续优化得到理想材料点 \(x^\*\) 后，需要它不要远离真实可购玻璃。可以加入 catalog-distance penalty：

$$
L_{\text{catalog}}(x)
=
\min_i
\left\|
W(x-x_i)
\right\|^2.
$$

为了可微，可以使用 soft-min：

$$
L_{\text{catalog}}(x)
=
-\tau
\log
\sum_i
\exp
\left(
-\frac{\|W(x-x_i)\|^2}{\tau}
\right).
$$

这里 \(W\) 用于归一化不同坐标的尺度，例如 \(n_d\)、\(V_d\)、\(\Delta P_{g,F}\) 的数量级不同。

这一步的意义非常关键：

> 优化器可以在连续玻璃空间中移动，但会被拉向真实 catalog 附近。

这正是该方法相比黑箱 AI 光学设计更容易连接真实生产的地方。

## 4. 投影回真实玻璃：连续优化不能作为终点

该方法不应声称直接产生最终可生产玻璃，而应采用如下流程：

1. 在连续材料空间中优化 \(x^\*\)；
2. 在真实 catalog 中寻找最近的 \(k\) 个玻璃；
3. 用厂家 Sellmeier 方程重新计算系统性能；
4. 比较连续 model glass 和真实玻璃投影后的性能差距。

流程可以写成：

$$
x^\*
\rightarrow
\text{top-k nearest catalog glasses}
\rightarrow
\text{Sellmeier verification}
\rightarrow
\text{manufacturing review}.
$$

这一步比单纯报告 \(n(\lambda)\) RMS 更有工程意义，因为它直接回答：

> 连续优化找到的材料方向是否能落回真实可购玻璃？

## 5. Côté 式最近真实玻璃量化

Côté et al. 的 differentiable lens design 使用了 quantized continuous glass variables：优化变量保持连续，但 forward pass 中把连续玻璃变量替换为最近的真实 catalog glass。这个思路可以直接和本文的三参数 Buchdahl 材料层结合。

区别是：

- Côté 的材料坐标主要是 \((n_d,V_d)\)；
- 本路线建议使用 \((n_d,V_d,\Delta P_{g,F})\)；
- Côté 重点解决 categorical glass selection；
- 本路线重点提供更物理的连续色散模型，再接入真实玻璃量化。

因此，两者不是替代关系，而是可以组合成：

$$
(n_d,V_d,\Delta P_{g,F})
\rightarrow
\text{continuous optimization}
\rightarrow
\text{nearest real catalog glass}
\rightarrow
\text{Sellmeier verification}.
$$

### 5.1 三参数最近玻璃距离

对真实玻璃目录中的每个玻璃 \(i\)，定义：

$$
x_i=(n_{d,i},V_{d,i},\Delta P_{g,F,i}).
$$

连续优化变量为：

$$
x=(n_d,V_d,\Delta P_{g,F}).
$$

最近玻璃可以定义为：

$$
i^\*
=
\arg\min_i
d_i(x)^2,
$$

其中：

$$
d_i(x)^2
=
\left(\frac{n_d-n_{d,i}}{\sigma_n}\right)^2
+
\left(\frac{V_d-V_{d,i}}{\sigma_V}\right)^2
+
\left(\frac{\Delta P_{g,F}-\Delta P_{g,F,i}}{\sigma_P}\right)^2.
$$

这里 \(\sigma_n,\sigma_V,\sigma_P\) 用来归一化三个坐标的尺度。这个三参数距离比只在 \((n_d,V_d)\) 平面找最近玻璃更适合 apochromat / secondary-spectrum 设计，因为 \(\Delta P_{g,F}\) 直接控制异常部分色散。

### 5.2 三种 catalog 结合模式

推荐把 catalog 结合分成三个强度等级。

| 模式 | Forward 使用 | 优点 | 风险 |
|---|---|---|---|
| Continuous Buchdahl | 连续 \(x\) | 梯度平滑、最容易优化 | 可能跑到不存在的材料 |
| Catalog-regularized Buchdahl | 连续 \(x\) + catalog-distance penalty | 仍然平滑，并靠近真实玻璃 | forward 仍是 model glass |
| Quantized real glass | 最近真实 catalog glass | 最接近生产流程 | nearest-neighbor 不可微，需要 straight-through estimator |

这三个模式可以逐步实现，不建议一开始就直接 hard quantization。

### 5.3 Soft catalog penalty

第一步建议使用软约束：

$$
L_{\text{catalog}}(x)
=
-\tau
\log
\sum_i
\exp
\left(
-\frac{d_i(x)^2}{\tau}
\right).
$$

该 penalty 让连续最优点靠近真实玻璃云，但不会破坏梯度连续性。它适合先接入 Notebook 03。

### 5.4 Hard quantization / straight-through estimator

当 soft penalty 与 top-k projection 已经跑通后，可以实现 Côté 式 hard quantization：

$$
x_q
=
x
+
\operatorname{stopgrad}(x_{i^\*}-x).
$$

forward pass 使用：

$$
x_q=x_{i^\*},
$$

但 backward pass 近似使用：

$$
\frac{\partial x_q}{\partial x}\approx I.
$$

这样优化器仍然能更新连续变量 \(x\)，但光学系统在 forward 中看到的是最近真实玻璃。

该模式最接近真实生产，但也最容易带来不稳定，因为最近玻璃索引 \(i^\*\) 会发生跳变。因此它应作为第三阶段，而不是第一阶段。

### 5.5 与 Buchdahl / Sellmeier 的关系

hard quantization 后有两种 forward 选择：

1. 使用最近真实玻璃的厂家 Sellmeier：

$$
n(\lambda)
=
n_{\text{Sellmeier},i^\*}(\lambda).
$$

2. 使用最近真实玻璃的三参数 Buchdahl surrogate：

$$
n(\lambda)
=
n_{\text{Buchdahl}}(\lambda;x_{i^\*}).
$$

第一种最接近最终验证；第二种保留和当前 Buchdahl pipeline 的一致性。更合理的实验顺序是：

1. continuous Buchdahl optimization；
2. top-k Sellmeier verification；
3. catalog-distance regularization；
4. hard quantized forward with Sellmeier。

### 5.6 相比 Côté 的具体创新点

如果引入最近真实玻璃量化，本文相对 Côté 类方法的差异可以写成：

> Côté et al. resolve the categorical nature of glass choice by quantizing continuous \((n_d,V_d)\) variables to the nearest catalog glass. This work extends the material coordinate itself by introducing \(\Delta P_{g,F}\) and an anchor-preserving Buchdahl surrogate, so the continuous glass layer preserves both primary dispersion and partial dispersion before catalog projection.

中文表述为：

> Côté 类方法解决了真实玻璃选择的离散性；本文进一步改进材料坐标和色散模型本身，用 \((n_d,V_d,\Delta P_{g,F})\) 和 anchor-preserving Buchdahl 保留主色散与部分色散，再接入最近真实玻璃量化。

因此，最有说服力的长期路线是：

$$
\text{Côté-style catalog quantization}
+
\text{three-parameter anchor-preserving Buchdahl material layer}.
$$

## 6. 热特性与 \(dn/dT\)

热特性也应该优先放入 Notebook 03 的系统级目标。

一种简单形式是：

$$
L_{\text{thermal}}
=
\sum_T
\left[
\text{BFL}(T)-\text{BFL}(T_0)
\right]^2.
$$

折射率随温度变化可以近似为：

$$
n(\lambda,T)
=
n(\lambda,T_0)
+
\frac{dn}{dT}(\lambda)(T-T_0).
$$

数据来源有两种：

1. 如果真实玻璃 catalog 提供 \(dn/dT\)，优先使用真实数据；
2. 如果没有，可以训练一个辅助 predictor：

$$
(n_d,V_d,\Delta P_{g,F})
\rightarrow
\frac{dn}{dT}.
$$

但 \(dn/dT\) 比 catalog-distance 更难，建议作为第二阶段扩展，而不是第一阶段主线。

## 7. 透过率

透过率本质上是波长相关函数：

$$
T(\lambda).
$$

它可以进入系统 throughput penalty：

$$
L_{\text{trans}}
=
\sum_\lambda
w_\lambda
\max(0,T_{\min}(\lambda)-T(\lambda))^2.
$$

但需要注意：透过率很难仅由 \((n_d,V_d,\Delta P_{g,F})\) 精确预测，因为它还依赖玻璃成分、杂质、厚度、厂家工艺和吸收边。

因此更实际的路线是：

- 当前阶段：先通过真实玻璃投影后检查 catalog transmittance；
- 未来阶段：如果进入 UV / NIR / broadband 系统，再考虑建立 \(T(\lambda)\) surrogate；
- 若使用 Notebook 05 的 NN residual 框架，可以扩展为同时预测 \(n(\lambda)\) 和 \(T(\lambda)\) 的高保真材料层。

## 8. 成本与供应风险

成本不适合直接作为连续物理量建模。更稳的做法是把成本视为真实 catalog 的 metadata。

假设每个真实玻璃有成本等级 \(c_i\)，连续材料点 \(x\) 附近的软玻璃权重为 \(p_i(x)\)，则可定义：

$$
L_{\text{cost}}(x)
=
\sum_i p_i(x)c_i.
$$

这里 \(p_i(x)\) 可以来自 soft nearest-glass 权重。

这样做的好处是：

- 成本来自真实 catalog，而不是任意拟合；
- 连续优化仍然可微；
- 优化器会倾向于靠近低成本、低供应风险的材料区域。

## 9. 加工限制

加工限制分成两类。

### 9.1 系统几何加工限制

这些应直接进入 Notebook 03 的 lens prescription 优化：

- 最小中心厚度；
- 最小边缘厚度；
- 最大曲率；
- 最小曲率半径；
- 最大 sag；
- 最大空气间隔；
- 最大系统总长；
- 最大口径；
- f-number 或 chief-ray 约束。

这类约束依赖镜头结构参数 \(\theta\)，可以写成：

$$
L_{\text{geometry-mfg}}(\theta).
$$

### 9.2 材料加工属性

这些更像真实玻璃 metadata：

- 硬度；
- 化学稳定性；
- 抛光难度；
- 最大可供应口径；
- 镀膜兼容性；
- 环境稳定性；
- 应力双折射风险。

它们可以通过 soft nearest-glass 权重进入：

$$
L_{\text{material-mfg}}(x)
=
\sum_i p_i(x)m_i.
$$

其中 \(m_i\) 是真实玻璃的加工难度或风险评分。

## 10. Notebook 05 的正确角色

Notebook 05 不应被表述为“替代 Notebook 03”。

更准确地说：

> Notebook 05 可以在未来替换 Notebook 03 中的 \(n(\lambda)\) evaluator。

Notebook 03 是系统级优化框架；Notebook 05 是更高保真的材料色散层。

当前 Notebook 03 使用：

$$
n_{\text{Buchdahl}}(\lambda;x).
$$

未来宽波段复杂系统可以替换为：

$$
n_{\text{total}}(\lambda;x)
=
n_{\text{Buchdahl}}(\lambda;x)
+
\varepsilon q(\lambda)\tanh(NN(x,\lambda)).
$$

其中 \(q(\lambda)\) 在 d/F/C/g 锚点为零，因此 neural residual 不破坏 anchor-preserving 结构。

Notebook 05 的优势是：

- 提高 off-anchor broadband \(n(\lambda)\) 保真度；
- 减少连续 surrogate 与 Sellmeier verification 之间的 gap；
- 更适合未来 VIS-NIR、多波长、复杂系统。

但它不直接解决：

- 可购性；
- 成本；
- 加工限制；
- 真实玻璃投影；
- 热稳定性；
- 供应链风险。

这些仍然应该由 Notebook 03 的系统级约束处理。

## 11. 推荐研究路线

当前最稳的路线是：

```text
Notebook 01
    Build anchor-preserving Buchdahl continuous glass model

Notebook 02
    Validate glass substitution and visible achromat/apochromat behavior

Notebook 03
    Use the model in differentiable optical design
    Add catalog-distance and production-aware constraints

Appendix / Future
    Notebook 04: MLP predictor is unnecessary
    Notebook 05: higher-fidelity broadband material layer
```

更具体地说，建议优先做以下三个实验。

### 实验 1：无 catalog penalty vs 有 catalog penalty

目标：

> 证明 catalog-distance penalty 能把连续最优点拉回真实玻璃附近。

对比：

- 只优化 optical merit；
- 优化 optical merit + catalog-distance penalty。

观察：

- 连续最优点到最近真实玻璃的距离；
- 光学性能是否明显下降；
- 最优点是否更容易投影回真实 catalog。

### 实验 2：连续最优 vs 最近真实玻璃 Sellmeier 复核

目标：

> 证明连续优化不是虚构结果，而是能有效指导真实玻璃选择。

流程：

1. 连续优化得到 \(x^\*\)；
2. 找最近的 top-k 真实玻璃；
3. 用真实 Sellmeier 重算；
4. 比较连续模型结果和真实玻璃结果。

这一步直接支撑实际生产结合。

### 实验 3：top-k real glass projection

目标：

> 证明连续模型能缩小离散玻璃搜索空间。

方法：

- 对连续最优点附近的 5 或 10 个真实玻璃重算系统；
- 与全 catalog brute-force 搜索对比；
- 检查 top-k 投影是否包含接近全局最优的真实玻璃。

如果成立，就可以说明：

> 连续 glass surrogate 可以作为真实玻璃替换的 search accelerator。

## 12. 建议论文表述

推荐把核心贡献写成：

> 本文提出一种 anchor-preserving Buchdahl continuous glass surrogate，将离散玻璃目录松弛为由 \((n_d,V_d,\Delta P_{g,F})\) 参数化的连续材料空间。该表示保留经典玻璃目录锚点，能够进入可微光学设计，并可通过 catalog-distance penalty 和 projection-to-real-glass verification 与真实可购玻璃连接。

英文可写为：

> The proposed method is not intended to replace manufacturer Sellmeier equations for known glasses. Instead, it provides a production-aware continuous relaxation of the discrete glass catalog. The optimized model-glass coordinates remain interpretable, can be constrained toward purchasable materials, and can be projected back to real catalog glasses for final Sellmeier verification.

## 13. 总结

当前项目最有工程意义的方向是：

$$
\text{continuous material space}
\rightarrow
\text{differentiable optical design}
\rightarrow
\text{catalog projection}
\rightarrow
\text{real Sellmeier verification}.
$$

加工性能和生产约束应该优先接入 Notebook 03，因为它们是系统级设计决策。

Notebook 05 的 neural residual 更适合作为未来宽波段复杂系统的材料层升级，而不是当前生产约束主线的核心。

一句话总结：

> 先用 Notebook 01 + Notebook 03 建立可生产约束的连续玻璃优化框架；未来如果进入宽波段复杂系统，再把 Notebook 05 的 Buchdahl+NN residual 作为更高保真的 \(n(\lambda)\) evaluator 插入同一个 Notebook 03 框架。
