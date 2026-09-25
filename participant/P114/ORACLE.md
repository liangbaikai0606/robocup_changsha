# P114 全局 Oracle（仅本地分析）

这个脚本**不能作为正式提交策略**。它通过训练环境的 `env.state()` 读取全局状态，违反正式执行时“每台机器人只能看局部观测”的限制。

用途只有一个：估计公开套件的可达上界，并判断当前约 216.67 分的规则策略到底还剩多少优化空间。

## 运行

从仓库根目录执行：

```bash
python participant/P114/oracle_eval.py --mode all
```

若要查看每一步的 assignment、coverage、collision：

```bash
python participant/P114/oracle_eval.py --mode predictive --trace
```

## 两个版本

- `greedy`：全局视角下按“当前距离到覆盖圈”的代价做 3! 枚举分配。
- `predictive`：利用全局状态里当前目标速度，估计每个 robot-target pair 的最早进圈时间，再做 3! 枚举分配，并用轻量预测追踪和制动控制。

这里的 predictive oracle **仍然不是数学意义的最优控制上界**：它不知道未来随机转向事件，也没有穷举 10 步联合动作。因此应把它理解为“强全局参考控制器”。如果它明显高于当前策略，再考虑做更强的 short-horizon MPC/beam oracle。

## 解读结果

例如：

- 当前策略 216.67，predictive oracle 250：规则策略已经很接近可达 ceiling。
- 当前策略 216.67，predictive oracle 350+：仍有明显的分配、预测或控制空间。

先看 oracle gap，再决定是否值得引入 RL。
