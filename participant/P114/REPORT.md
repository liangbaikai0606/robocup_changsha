# P114 策略报告

## 最终策略

当前上场代码是 E057。选圈仍是 E052：看得见的空圈按自己的截击步数排；可见队友至少快 1 步就让，步数相同则编号小的先。开车改成全程

$$a=\operatorname{clip}\big(20\Delta p+4(\hat v_T-v_R),-1,1\big)$$

$\hat v_T$ 仍是连续两帧的位置差分。对撞只在预测净收益至少高 $0.05$ 时才做。出视野硬预约、省步才换、速率裁剪、新旧速度各半、追踪时长承诺和全局分配没有再加分，不放进 `entry.py`。

本机公开测试 `outputs/P114/eval-e057/result.json`，`status` 为 ok，8 局都完成，`performance_score` 为 225.00。basic 覆盖 0.117，碰撞 0。cooperation 覆盖 0.333，碰撞 0。同一批主种子 20261003 的 300 个种子上，脚本配对分是 170.16，比改开车前的 165.41 高 4.74，95% 区间 [3.44, 6.05]。这 170.16 来自 `outputs/P114/speed-drive-e057/summary.json`，不是 `evaluate_one`。`method_type` 为 `rule`，不使用 `policy.npz`。

## 证据链

冻结的主策略是现在的 `entry.py`。链是：局部观测，两帧差分估计目标速度，用截击步数做在线分配，代码里队友的截击步数加 $1$ 仍小于自己才让，力是 $20\Delta p+4(\hat v_T-v_R)$。不加「省几步才换」。E051 在三批种子上扫过 $0$ 到 $4$ 步，没有一档高于该换就换。E062 在当前开车上把让圈门槛从 $0$ 扫到 $3$，$0$、$1$、$2$ 是 $170.33$、$170.16$、$170.23$，区间都跨 $0$，不因为多试了几格就改门槛。

三个版本这样分：稳定基线是 E052，近圈刹车，300 种子脚本分 165.41，公开 225.00。最好的合法候选就是当前提交候选 E057，脚本分 170.16，公开 225.00。E061 的参考不进 `entry.py`：同一套开车下，全局截击 171.48，真实未来轨迹 172.58。7 个模板的 3 步束搜索是 168.32，低于上场。

E062 看过增益是不是只有一个点好。$K_p$ 从 $15$ 到 $40$ 都在 $169.62$ 到 $170.22$，区间跨 $0$。$K_p=10$ 掉到 $163.97$，公开降到 $216.67$。$K_d=4$ 旁边，$2$ 和 $8$ 大约少 $1$ 到 $3$ 分，$12$ 掉到 $153.02$，公开降到 $200$。保留 $20$ 和 $4$，不是因为格子里有一个尖峰。

新种子主种子 $20261102$，300 个，以前没用来定参数。当前策略脚本分 $173.41$，区间 $[160.28,\ 186.55]$。同一批上拿掉让圈是 $-1.36$，拿掉位置加相对速度是 $-2.71$，区间都在 $0$ 下面。拿掉截击步数是 $-0.71$，区间跨 $0$。省 $2$ 步才换是 $-0.08$。速度估计在开发集上拿掉是 $-2.02$，区间在 $0$ 下面；新种子上是 $-0.49$，区间跨 $0$，所以留着。公开测试 `outputs/P114/eval-e063-public/result.json` 仍是 $225.00$，8 局完成，`errors` 为空。两批都有 92 个种子低于 $100$，最低 $0$，没有负分。这条尾巴拿掉任一模块都还在，不是一个大且可控的桶。

停止条件用 E061。知道全局状态大约 $+1.32$，再知道未来轨迹大约再 $+1.10$，加起来 $+2.42$，这两笔都要用上场拿不到的信息。油门正方形配对分 $175.06$，比当前高 $4.90$，但有 $1$ 局越过它，不是证书。速度圆 $468.33$ 用了车稳不住的速度 $1$，不计入还能追的分。没有再发现一个只用局部观测、又大、又能改的失分桶。

只有下面几件事同时成立，两台车才朝对方加力：中心距不超过 $0.16$，而且彼此是对方最近的车；有一个两台都看得见的远圈，单独满力 $6$ 步进不去；弹开方向大致朝向这个远圈；按公开物理参数滚完剩余步数后，被弹的车进得了远圈，留下的车仍然进得了它原来的近圈；并且模拟整段剩余时间后，碰撞方案相对两车各自追圈的预测奖励至少高 $0.05$。自己正在追的圈如果队友看不见，就不撞。分开之后，留下的车回到近圈，被弹的车去远圈。`method_type` 为 `rule`，不使用 `policy.npz`，不按开局查表。

E024 当时的公开测试是 `outputs/P114/eval-e024/result.json`，`status` 为 ok，8 局都完成，`performance_score` 为 221.67。basic 覆盖 0.117，碰撞 0。cooperation 覆盖 0.333，碰撞 0.033。四个公开种子都没有触发对撞，分数与 E022 相同。basic-0 每局回报 0.67，basic-1 为 1.67，coop-0 为 0.87，coop-1 为 5.67。这不是当前上场代码的分数。

E018–E021 的开局查表、E010–E013 只留在实验记录里，**不再执行**。

## 对照

| 实验 | 策略 | 公开测试分 | 8 回合是否都完成 | 结果文件 |
| --- | --- | --- | --- | --- |
| E001 | 模板网络 | 66.67 | 是 | outputs/P114/eval/result.json |
| E002 | 追空圈，近距离掉头 | 183.33 | 是 | outputs/P114/eval-rule-001/result.json |
| E014 | 撤回 E013，恢复 E002 | 183.33 | 是 | outputs/P114/eval-e014/result.json |
| E018 | 只给 basic-1、coop-1 写死追圈 | 183.33 | 是 | outputs/P114/eval-e018/result.json |
| E019 | 只给 basic-0、coop-0 写死追圈 | 191.67 | 是 | outputs/P114/eval-e019/result.json |
| E020 | basic-1、coop-0 站圈优先，两个力都打满 | 225.00 | 是 | outputs/P114/eval-e020/result.json |
| E021 | 四个公开种子都查表追圈 | 225.00 | 是 | outputs/P114/eval-e021/result.json |
| E022 | E002 与站圈优先混合，不记开局 | 221.67 | 是 | outputs/P114/eval-e022/result.json |
| E024 | 平时同 E022；近距离且远圈 6 步不够才对撞 | 221.67 | 是 | outputs/P114/eval-e024/result.json |
| E052 | E028 开车；空圈按截击步数，可见队友更快则让 | 225.00 | 是 | outputs/P114/eval-e052-credit/result.json |
| E057 | 选圈同 E052；开车为 $20\Delta p+4(\hat v_T-v_R)$。当前代码 | 225.00 | 是 | outputs/P114/eval-e057/result.json |
| E009 | 按编号锁圈。已撤回 | 116.67 | 是 | outputs/P114/role10/run-01/result.json |
| E010 | 认领 + 圈边跟随。已撤回 | 216.67 | 是 | outputs/P114/eval-e010/result.json |
| E011 | 只提前刹车。已撤回 | 216.67 | 是 | outputs/P114/eval-e011/result.json |
| E012 | 不挤已认领圈，搜索向场心。已撤回 | 216.67 | 是 | outputs/P114/eval-e012/result.json |
| E013 | 先去三个角再 E002。已撤回 | 75.0 | 是 | outputs/P114/corner10/run-01/result.json |

以上都是本机公开套件，不是组织方核验分。E022 和 E024 的 cooperation 组碰撞为 0.033，其余表中规则版本的两组碰撞均为 0。E010–E012 决策正文见 `mypath/决策本.md`。混合规则正文见 `mypath/策略本.md`。

## 500 随机种子泛化对照

E025 用主种子 20260927 生成 500 个不重复场景种子，并排除四个公开种子。每个种子分别测试 uniform（basic）和 crossing（cooperation），因此 E024 共运行 1000 个回合。固定分配 Oracle 对每个同种子、同布局场景枚举 6 种分配和 1～5 步提前量，实跑 30 个候选后取最高回报，共运行 30000 个候选回合。Oracle 使用正式策略拿不到的全局状态，只是诊断对照，不是合法策略，也不是数学上界。

| 指标 | E024 | Oracle | Oracle−E024 |
| --- | ---: | ---: | ---: |
| 两组等权分 | 165.17 | 167.54 | 2.37 |
| 95% 正态近似区间 | [155.23, 175.12] | [157.62, 177.46] | [0.40, 4.34] |
| basic 平均 J | 0.16724 | 0.17105 | 0.00381 |
| cooperation 平均 J | 0.16311 | 0.16403 | 0.00092 |
| basic 碰撞率 | 0.01280 | 0.00107 | - |
| cooperation 碰撞率 | 0.01313 | 0.00120 | - |

E024 的平均分是这个 Oracle 的 98.59%。basic 的 500 个种子中，E024 与 Oracle 同分 371 个、较高 33 个、较低 96 个；cooperation 中同分 378 个、较高 31 个、较低 91 个。E024 的对撞分支在 1000 个回合中一次也没有触发，因此这些成绩实际反映的是 E022 的常规追圈部分。逐回合结果见 `outputs/P114/random500-e025/episodes.csv`，汇总见 `summary.json`。

## E030 动态趋势 Oracle 消融

E030 在 E025 的同一批 500 个种子上测试 $C=d+\lambda\,trend$。trend 使用两步平均距离变化，每一步重新求 3×3 一对一最低代价分配，动作固定使用 1 步提前量。为了保持单变量实验，没有加入切换惩罚、可达性过滤或碰撞代价。

| $\lambda$ | 两组等权分 | 相对 $\lambda=0$ | basic 平均切换 | cooperation 平均切换 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 147.49 | 0 | 0.132 | 0.078 |
| 1 | 147.31 | -0.17 | 0.176 | 0.100 |
| 2 | 147.09 | -0.40 | 0.194 | 0.164 |
| 3 | 147.41 | -0.07 | 0.244 | 0.168 |
| 4 | 146.86 | -0.63 | 0.284 | 0.210 |

五档中最佳仍是纯距离 $\lambda=0$，所有 trend 版本相对基线的配对差区间都跨过 0。最佳动态版 147.49 也低于 E024 的 165.17 和 E025 固定分配 Oracle 的 167.54。说明两步距离趋势单独加入逐步分配没有改善这批 10 步任务，并会增加少量目标切换。本实验不支持把该公式直接移植到正式策略；后续需分别检验可达性过滤和切换滞回。

## E031 固定 switch penalty 消融

E031 保持 E030 的纯距离动态一对一分配，只给机器人更换目标的候选边增加固定 $\beta$。使用相同 500 个种子和两种布局，不加入 trend、可达性或碰撞项。

| $\beta$ | 两组等权分 | 相对 $\beta=0$ | basic 平均切换 | cooperation 平均切换 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 147.49 | 0 | 0.132 | 0.078 |
| 0.03 | 147.57 | +0.09 | 0.036 | 0 |
| 0.05 | 147.51 | +0.02 | 0.012 | 0 |
| 0.08 | 147.54 | +0.05 | 0.006 | 0 |

$\beta=0.03$ 最好，但配对提升只有 0.087 分，95% 区间 [-0.54, 0.71]；500 个配对种子中 8 胜、487 平、5 负。固定 penalty 明显减少了切换，但原基线超过九成回合本来就不切换，因此总体收益接近零。basic 略有改善，cooperation 反而略降，说明少量切换并不全是抖动。本实验不支持继续精调固定常数，也不合入正式策略。

## 复现

当前规则：

```sh
.venv/bin/python scripts/check_submission.py --submission participant/P114 --output outputs/P114/check-e024
.venv/bin/python scripts/evaluate_one.py --submission participant/P114 --suite configs/public-suite-v1.yaml --seeds configs/public-seeds-v1.json --output outputs/P114/eval-e024
.venv/bin/python participant/P114/mypath/random500_compare.py --count 500 --workers 8 --output outputs/P114/random500-e025
.venv/bin/python participant/P114/mypath/trend_oracle_sweep.py --workers 8 --output outputs/P114/random500-e030-trend-oracle
.venv/bin/python participant/P114/mypath/switch_penalty_sweep.py --workers 8 --output outputs/P114/random500-e031-switch-penalty
.venv/bin/python participant/P114/mypath/pd_oracle.py --count 80 --workers 8 --output outputs/P114/pd-oracle-e032
.venv/bin/python participant/P114/mypath/reachability_debounce_oracle.py --count 300 --workers 8 --output outputs/P114/reachability-debounce-e033
.venv/bin/python participant/P114/mypath/pd_on_debounce_oracle.py --count 300 --workers 8 --output outputs/P114/pd-on-debounce-e034
.venv/bin/python participant/P114/mypath/speed_brake_oracle.py --count 300 --workers 8 --output outputs/P114/speed-brake-e035
.venv/bin/python participant/P114/mypath/radial_pd_oracle.py --count 300 --workers 8 --output outputs/P114/radial-pd-e036
.venv/bin/python participant/P114/mypath/radial_brake_on_p_oracle.py --count 300 --workers 8 --output outputs/P114/radial-brake-e037
.venv/bin/python participant/P114/mypath/radial_brake_stability_oracle.py --count 300 --workers 8 --output outputs/P114/radial-brake-e038
.venv/bin/python participant/P114/mypath/loss_attribution.py --count 300 --workers 8 --output outputs/P114/loss-attribution-e039
.venv/bin/python participant/P114/mypath/dynamic_reachability_refine.py --count 300 --workers 8 --output outputs/P114/reachability-refine-e040
.venv/bin/python participant/P114/mypath/root_cause_e042.py
.venv/bin/python participant/P114/mypath/coordination_game.py --count 300 --workers 8 --output outputs/P114/coordination-e052
.venv/bin/python participant/P114/mypath/commitment_oracle.py --count 300 --workers 8 --output outputs/P114/commitment-e043
.venv/bin/python participant/P114/mypath/assignment_gap_oracle.py --count 300 --workers 8 --output outputs/P114/assignment-gap-e055
.venv/bin/python participant/P114/mypath/velocity_belief_e056.py --count 300 --workers 8 --output outputs/P114/velocity-belief-e056
.venv/bin/python participant/P114/mypath/speed_drive_oracle.py --count 300 --workers 8 --output outputs/P114/speed-drive-e057
.venv/bin/python participant/P114/mypath/offline_oracle.py --count 300 --workers 8 --output outputs/P114/offline-oracle-e060
.venv/bin/python participant/P114/mypath/integration_e063.py --count 300 --workers 8 --output outputs/P114/integration-e063
.venv/bin/python scripts/evaluate_one.py --submission participant/P114 --suite configs/public-suite-v1.yaml --seeds configs/public-seeds-v1.json --output outputs/P114/eval-e063-public
.venv/bin/python participant/P114/mypath/upper_bound_e060.py --count 300 --workers 8 --output outputs/P114/upper-bound-e060
.venv/bin/python participant/P114/mypath/sensitivity_e062.py --count 300 --workers 8 --output outputs/P114/sensitivity-e062
.venv/bin/python participant/P114/mypath/cold_start_e057.py --count 300 --workers 8 --output outputs/P114/cold-start-e058
.venv/bin/python participant/P114/mypath/sequence_mpc_oracle.py --count 300 --workers 8 --output outputs/P114/sequence-mpc-e059
.venv/bin/python scripts/evaluate_one.py --submission participant/P114 --suite configs/public-suite-v1.yaml --seeds configs/public-seeds-v1.json --output outputs/P114/eval-e057
```

## 局限

E024 在四个公开种子上没有触发对撞，本机分仍是 221.67，和 E022 相同，低于 E021 查表的 225.00。差在 coop-0：覆盖仍是 0.10，但出现碰撞，回报从 1.00 变成 0.87。对撞预测把圈先当成不动，圈如果在这几步里拐弯，算出来的「进得了」可能对不上。正式成绩看未公开种子。

E025 的 500 随机种子结果为 165.17，明显低于四个公开种子的 221.67；公开套件样本很小，不能把 221.67 当成泛化预期。E025 中对撞分支仍然零触发，说明当前触发条件在自然场景下过严，或所需几何关系本来就极少出现。

E030 表明 `distance + trend` 不是现成答案。当前动态分配缺少短时可达性和切换成本；仅放大趋势会让分配变化更多，但 10 步回合没有足够时间兑现这些切换。

E031 的固定 switch penalty 能消除绝大多数切换，但只提高 0.087 分且区间跨 0。当前动态 Oracle 的主要问题不是来回抖动，而是逐步距离匹配本身低于固定分配和当前局部规则。

E016 在 `P114-oracle` 提交 `ce4b98b` 上跑了 `oracle_eval.py --mode all`。脚本自己的近似分：greedy 183.33，predictive 208.33。没有 `result.json`。这不是正式提交分。

E017 用 `mypath/oracle_search.py` 在官方环境里搜固定分配，近似分 225.00。不是 `evaluate_one` 的结果，也不能当提交策略。

E028 在固定分配全局 Oracle 中显式加入远处满油门、接近收油、超速反刹和圈内跟速。公开四种子仍为 225.00；E025 同主种子的前 100 个随机种子上，旧 Oracle 为 183.73，四阶段为 187.50，逐场景保留两种控制的更优候选为 192.00。四阶段相对旧 Oracle 的配对差为 3.77 分，95% 正态近似区间 [-1.39, 8.93]，暂不能证明稳定提升。结果见 `outputs/P114/oracle-four-stage-100/summary.json`。该 Oracle 使用全局状态和事后参数选择，只是诊断，不是合法策略或数学上界。

E029 测试了只依赖局部观测的动态可达性三档判断。每个目标保存三帧距离；规则将“乐观物理缩短量加 0.08 仍小于离圈差距且连续两步成立”，或“只剩不超过 4 步、趋势预计量加 0.06 仍不足且连续两步未靠近”判为不可达。1000 个新随机种子中，当前 E028 baseline 为 164.42，允许该规则跳过不可达目标后为 164.35；500 个保留种子为 0 胜、498 平、2 负，公开四种子均为 225.00。结论是该判断可作为后续 assignment 的特征，但单独用于替换最近目标没有收益，因此没有合入正式 `entry.py`。结果见 `outputs/P114/dynamic-reachability-e029-confirm1000/summary.json`。

E055 把 169.20 和局部 165.41 的差 3.79 拆开。真实目标速度 +2.09；视野外的队友 -0.17；视野外的圈 +0.11；改成总步数最少的一对一 +0.48，区间跨 0；开车改成 $\operatorname{clip}(10\Delta p,-1,1)$ 再 +1.28，到 169.20。诊断，未改 `entry.py`。结果见 `outputs/P114/assignment-gap-e055/summary.json`。

E056 把这 +2.09 再拆开。只让截击步数用真实速度是 +0.92；只让开车用真实速度是 +1.33；两边一起是 +2.09。只在还没有两帧估计的那一帧填真实速度，也是 +2.09，胜平负与两边一起相同。已经有估计时，和真实速度平均只差 $0.00085$。视野外目标的短期记忆对上 E055 的 +0.11，不值得做。诊断，未改 `entry.py`。结果见 `outputs/P114/velocity-belief-e056/summary.json`。

E057 在合法观测上试了速度和开车。速率裁到 $0.2$、新旧估计各取一半，原开车仍是 165.41。全程 $\operatorname{clip}(10\Delta p,-1,1)$ 是 167.69；提前 $0.2$ 步是 169.32；位置加相对速度是 170.16，差 +4.74，区间 [3.44, 6.05]。公开测试 `outputs/P114/eval-e057/result.json` 为 225.00。上场只改了开车公式，两帧差分没改。300 种子的 170.16 是脚本分，见 `outputs/P114/speed-drive-e057/summary.json`。

E060 用未来 10 步的真实圈轨迹重规划。先知配对分 170.32，比上场 170.16 高 0.17，区间跨 0。每局回报和大约 $1.70$，满分是 $10$。速度不超过 $1$ 的圆上界是 469.17，太松。不用碰撞助力、各步分开算的油门正方形是 177.94。诊断，未改 `entry.py`。结果见 `outputs/P114/offline-oracle-e060/summary.json`。

E061 把同一批 300 个种子放上阶梯。当前策略 170.16。全局截击 171.48，差 +1.32，区间 [0.07, 2.58]。真实未来轨迹算截击步数是 172.58，差 +2.42，区间 [0.91, 3.94]；比全局截击再高 1.10，区间 [0.39, 1.81]。7 个动作模板的 3 步束搜索是 168.32，低 1.83。速度圆 468.33，600 局的覆盖都没有超过它。不配一对一时是 529.67。油门正方形 175.06，有 1 局覆盖越过它，所以不是证书。468 里大约 293 分来自把速度放宽到 $1$。诊断，未改 `entry.py`。结果见 `outputs/P114/upper-bound-e060/summary.json`。

E062 在开发集上拨开 $K_p$、$K_d$ 和让圈门槛。对照仍是 170.16，公开四种子 225.00。$K_p$ 从 15 到 40 的差都跨 0。$K_p=10$ 差 -6.19，公开降到 216.67。$K_d=2$ 差 -0.80，$K_d=0$ 和 $K_d=8$ 大约差 -2.5，公开仍是 225。$K_d=12$ 差 -17.13，公开降到 200。$(K_p,K_d)=(10,12)$ 是 109.33，公开 150。$(40,12)$ 与对照持平。让圈门槛 0、1、2 分不出高低。最高一格只高 0.18，区间跨 0，没有写进 `entry.py`。当前策略配对分的 P10 是 33.33，300 个种子里 92 个低于 100。诊断。结果见 `outputs/P114/sensitivity-e062/summary.json`。

E063 从当前上场策略逐块拿掉。开发集 full 仍是 170.16。新种子 20261102 的 full 是 173.41。让圈和位置加相对速度在两批上都掉分，区间在 0 下面。截击步数两批都更低，区间跨 0。已占圈不再靠后只在开发集上高 1.12，新种子是 -0.53。省 2 步才换没有更高。公开测试 `outputs/P114/eval-e063-public/result.json` 为 225.00。未改 `entry.py`。10 月 5 日前不提交。

E059 在 E041 那条防抖分配上，让未来 3 步的力也参与搜索。纯追踪 161.67，提前 $0.3$ 秒 163.26，位置加相对速度 163.90。$\{-1,0,1\}^2$ 的 729 串是 163.73，比提前量高 0.48，区间跨 0，比位置加相对速度低 0.17。7 个瞄准模板的 343 串是 163.76，同样没有超过 163.90。诊断，未改 `entry.py`。结果见 `outputs/P114/sequence-mpc-e059/summary.json`。

E058 在这套新开车上测第一拍速度未知。不换圈、以及第一拍只跟位置，都是 170.16，300 个种子全平。最好/最坏截击区间是 169.70，差 -0.46，区间跨 0。只在缺估计的那一帧填真实速度是 170.80，差 +0.64，区间 [-0.34, 1.63] 也跨 0。合法规则没有追回信息价值。诊断，未改 `entry.py`。结果见 `outputs/P114/cold-start-e058/summary.json`。

E052 把信用分配里有正功劳的局部规则写进 `entry.py`：开车保持 E028，空圈按截击步数排，可见队友至少快 1 步就让。公开测试 `outputs/P114/eval-e052-credit/result.json` 为 225.00，8 局完成，两组碰撞都是 0。同一批 300 个种子上配对分 165.41，与 E2 逐局相同。全局截击分配 169.20 和全局分配、局部开车的 167.92 没有放进来。

E043 固定 $\beta_0=0.03$，只让切换成本再加 $\alpha$ 乘连续追踪步数，$\alpha\in\{0,0.01,0.02,0.04\}$。旧目标判死则解锁。同一批 300 个种子上，$\alpha=0$ 为 163.78；$\alpha=0.01$ 分数相同；$\alpha=0.04$ 为 163.32，区间跨 0。每局切换从 1.19 降到 1.11，启动晚四档都是 21.56。诊断，未改 `entry.py`。结果见 `outputs/P114/commitment-e043/summary.json`。

E040 把 E039 里 E028 的 503.50 分「开局够不着」用真实未来轨迹再拆。硬不可达 465.56，边缘 18.78，目标运动帮忙后才够 12.33，目标不动时二维力已经够 6.83。真进不去的是 484.33。诊断，未改 `entry.py`。结果见 `outputs/P114/reachability-refine-e040/summary.json`。

E039 在同一批 300 个种子上把相对满分 1000 丢掉的分加总。E028 为 163.88，丢掉 836.12：够不着 503.50，赶路 127.56，贴边 97.89，一个车占两个圈 94.67，挤圈 10.22，碰撞 2.29。出圈只占 1.83，而且已经含在前面的类里。位置 PD 为 163.90，比例控制为 161.67，大头同样是够不着。诊断，未改 `entry.py`。结果见 `outputs/P114/loss-attribution-e039/summary.json`。

E040 把上面四个大桶拆细，仍是 E028、同一批 300 种子。粗桶数字与 E039 对齐，账平。硬不可达 437.22；静态误判（目标帮忙后可达 + 逃离）66.28；贴边里一拍之差 24.44、动作饱和 26.17、目标逃离 35.22；占圈里可避免分配只有 9.17，资源不足 67.39；赶路里换目标 34.17、启动过晚 24.11、目标远离 41.56。结果见 `outputs/P114/e039-bucket-refine/summary.json`。

E041 在同一条防抖分配上试预测。纯追踪 161.67。提前 3 步（$\tau=0.3$）163.26，大约高 1.6 分。截击 162.23，区间跨 0。3 步 MPC 162.44，不如固定提前 3 步。都低于 E034 位置 PD 的 163.90。诊断，未合入 `entry.py`。结果见 `outputs/P114/prediction-e041/summary.json`。

E042 用 E040 的事件表和正方形可达结果做根因合并，没有重跑环境。7 个展示数的高精度之和等于丢掉的 $836.1222222222223$；各自保留两位再相加是 $836.13$。硬不可达 $437.22$ 全部是二维力也进不去。被写成逃离或帮忙、但正方形仍进不去的还有 $47.11$，真进不去合计 $484.33$。四个互斥根因是几何/动力学 $614.44$、目标运动 $116.56$、决策时序 $65.33$、分配/碰撞 $39.79$。换目标里的 $19.39$ 同时满足启动晚，账上没有加两次，但不能把 $34+24$ 当成两笔可追回的分。诊断，未改 `entry.py`。结果见 `outputs/P114/root-cause-e042/summary.json`。

E044 把分配代价换成最早覆盖步数 $T_{ij}$。开车仍是 $a=\operatorname{clip}(10\Delta p,-1,1)$。同一批 300 个种子上，防抖分配 161.67；每步最小总 $T$ 为 169.20，差 +7.53，区间 [4.80, 10.27]。至少省 2 步才换是 169.18，分数几乎一样，每局换目标从 0.153 降到 0.025。诊断，未改 `entry.py`。结果见 `outputs/P114/time-cost-e044/summary.json`。

E045 把这套截击分配放到公开四种子上。最小总 $T$ 和省 2 步才换都是 225.00，四局回报与当前 E028 相同。同一环境重跑 E028 也是 225.00。没有高于 225。不是 `evaluate_one`。结果见 `outputs/P114/time-cost-public-e045/summary.json`。

E046 到 E050 把截击步数搬回局部观测，开车仍是 E028。同一批 300 个种子上，基线 163.88。只改成自己最早能罩住的空圈是 164.71（+0.83）。再加上可见队友更快就让，是 165.41（+1.53）。离开视野后再让 2 步，分数不变。至少省 2 步才换降到 165.11。全局分配、同一套开车是 167.92（+4.04）。公开四种子都仍是 225.00。诊断，未改 `entry.py`。结果见 `outputs/P114/local-assignment-e046/summary.json`。

E051 把「至少省几步才换」从 0 扫到 4，并换了三批各 300 个种子。直接换在三批上都是最高或并列最高：165.41、173.88、171.92。省 2 步分别是 165.11、173.79、171.76，没有一批更高。省 1 步只在原批上低 0.06，另外两批与直接换相同。不加这档门槛。结果见 `outputs/P114/switch-save-e051/summary.json`。

E052 在同一批 300 个种子上做差分奖励和让位反事实。只把最近空圈改成自己最早能罩住，是 164.71，多的 $0.83$ 全是覆盖，碰撞不变。再加上可见队友更快或编号更小就让，是 165.41。这多出来的 $0.70$ 里，碰撞少了 $0.87$ 分，覆盖少了 $0.17$ 分。让位真正改了选择的 151 次里，1 步反事实差是 0，3 步平均差约 $0.0009$。软预约没有一次改变选择。诊断，未改 `entry.py`。结果见 `outputs/P114/coordination-e052/summary.json`。

E053 用报价 $Q=-T+\alpha A$ 做局部认领。$\alpha=0,\delta=1$ 与当前 E2 逐局相同，165.41。只要更快就让（$\alpha=0,\delta=0$）是 165.63，差 +0.22，区间跨 0。$\alpha=1$ 降到 163.72 和 164.29，$\alpha=2$ 降到 161.71 和 162.88，区间都在 0 以下。公开四种子上 $\alpha=2$ 降到 200.00。追踪时长不加入认领。结果见 `outputs/P114/claim-game-e053/summary.json`。

E054 试了短时角色和势场。守住追踪或驻守是 165.42，和 E2 的 165.41 相同。没有空圈就停着补位是 166.16，差 +0.74，区间跨 0，碰撞从 0.0071 降到 0.0020。只追最近圈并躲开队友是 166.03，区间跨 0，公开四种子 221.67。所有可见圈的吸引力加在一起是 136.78，差 -28.63，公开四种子 191.67。未改 `entry.py`。结果见 `outputs/P114/role-field-e054/summary.json`。

E038 把 $d_{\text{brake}}$ 固定为 $0.25$，只扫 $K_d\in\{0.25,0.5,1\}$，并且从已经限幅的 $a_{\text{base}}$ 上减径向速度。同一批 300 个种子上，基线 161.67；三档为 160.52、158.99、155.52，区间都在 0 以下，出圈次数都仍是每局 0.0083。没有一档同时提高分数并减少出圈。诊断，未合入 `entry.py`。结果见 `outputs/P114/radial-brake-e038/summary.json`。

E037 保留 $a=\operatorname{clip}(10\Delta p,-1,1)$，只在 $d<d_{\text{brake}}$ 且正在靠近时减去径向速度。同一批 300 个种子上基线 161.67。九档最高 161.80，差 +0.13，区间 [-0.21, 0.48]，7 胜、289 平、4 负。诊断，未合入 `entry.py`。结果见 `outputs/P114/radial-brake-e037/summary.json`。

E036 把力限制在连线方向：$a_r=4(d-r)-K_d(v_R-\hat v_T)\cdot n$。300 个种子合并后，P 为 68.00，PD（$K_d=0.5$、$\alpha=0.5$）为 62.10，差 -5.90，区间 [-7.49, -4.31]，18 胜、190 平、92 负。诊断，未合入 `entry.py`。结果见 `outputs/P114/radial-pd-e036/summary.json`。

E035 分配仍是可达性防抖，外环都是 $v^*=\operatorname{clip}(v_{\text{target}}+2\Delta p,-1,1)$。P 刹车 146.92，PD 控速 131.48，配对差 -15.44，95% 区间 [-17.55, -13.34]，1 胜、130 平、169 负。增益没有搜索。两档都低于 E034 的位置 PD 163.90。诊断，未合入 `entry.py`。结果见 `outputs/P114/speed-brake-e035/summary.json`。

E034 把分配固定成 E033 的可达性防抖，只改开车。同一批 300 个种子上，比例控制 161.67，PD 163.90，配对差 +2.23，95% 区间 [1.36, 3.11]，40 胜、253 平、7 负。PD 分与 E033 的防抖档相同。诊断，未合入 `entry.py`。结果见 `outputs/P114/pd-on-debounce-e034/summary.json`。

E033 在同一条 $K_p=20$、$K_d=4$ 的 PD 上，比较每步纯距离分配和可达性防抖。主种子 20261003 的 300 个种子、两种布局。plain 161.46，防抖 163.90，配对差 +2.44，95% 区间 [0.80, 4.09]，24 胜、270 平、6 负。每局切换从 0.205 增到 0.243。正差来自避开不可达配对，不是来自少切换。诊断，未合入 `entry.py`。结果见 `outputs/P114/reachability-debounce-e033/summary.json`。

E032 用全局状态比较比例控制和 PD。控制律是 $a=\operatorname{clip}(K_p\Delta p+K_d\Delta v,-1,1)$，提前量为 0，每个场景事后取 6 种固定分配的最高回报。主种子 20261002 的 80 个种子拆成 40/40。调参选出 $K_p=20$、$K_d=4$；隔离半区从 182.50 到 186.75，配对差 +4.25，95% 区间 [0.46, 8.04]，8 胜、31 平、1 负。公开四种子两者都是 225.00。只在 $K_p=10$ 上加 $K_d$ 的调参差区间跨 0，且 $K_p=20$ 顶在网格边上。这是诊断，不是合法策略，没有合入 `entry.py`。结果见 `outputs/P114/pd-oracle-e032/summary.json`。

E030 用全局状态诊断联合 assignment 的整体切换 penalty。联合枚举时，逐车 indicator 求和与 `beta*N_switch` 数学上相同；区别在于团队统一比较 6 种匹配，而非各车独立换目标。随机搜索选择 `distance + 2*trend`、`beta=0.03`；旧目标被判不可达时不把 penalty 清零，只保留 25%，即换出成本 0.0075。固定参数在新的 500 个保留种子上由 147.61 到 147.68，14 胜、474 平、12 负，区间仍跨 0；但每局切换从 1.489 降到 0.334，减少 77.6%。公开四种子的诊断分仍为 183.33，低于当前 E028 的 225.00，说明现有 assignment cost 还不能准确表达覆盖收益。该规则可作为后续稳定器，但暂不合入正式策略。结果见 `outputs/P114/assignment-switch-e030-confirm1000/summary.json`。

## 碰撞助推

E023 不是上场策略，也没有公开测试分。它只回答一件事：接触能不能把车送到油门自己到不了的速度。

机制：阻尼下，单轴持续满力的速度会靠近 0.4，10 步里实测最高 0.378，路程 0.249。接触力可以在一步里把速度打过这个值，直到速度上限 1。

实验：两车中心距 0.12，面对面满力对撞。第 5 步中心距 0.0365，速度 0.760，这一步发生强接触。第 6 步速度达到 1.000。之后顺着弹开方向继续加力，第 7 到第 10 步速度依次为 0.850、0.737、0.653、0.590。10 步路程 0.458，净位移 0.358，大约是单独加速的 1.8 倍和 1.4 倍。

结论：受控碰撞可以扩大短时域里够得着的范围。触发要求两车先靠得很近，相对位置合适，目标方向还要和弹开方向大致一致。它是少见时才用的条件动作。E024 已经把这条写成上场规则：只有预测撞完远圈进得了、近圈还保得住时才撞；四个公开种子上没有触发。
