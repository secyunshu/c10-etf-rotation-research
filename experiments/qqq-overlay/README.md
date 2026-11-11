# V2增加条件QQQ底仓

这是2026-11-11用户确认的单一新方案，当前标准默认仍为原V2前两只。只研究2022–2025；未测试2026，不把已见历史称独立OOS。

|通过原防守门槛后的风险目标|轮动槽|额外QQQ|SHY|目标现金|
|---|---|---|---|---|
|包含QQQ|每槽40%|0|15%起，失败槽转入|5%|
|不含QQQ|每槽40%|固定15%|5%起，失败槽转入|0|

完整[冻结计划](PLAN.md)、[计划挑战](PLAN_REVIEW.json)、[最终结果](FINAL_REPORT.md)、[逐年表](results/ANNUAL.csv)、[完整汇总](results/SUMMARY.csv)。与原V2不同，额外QQQ不受动量防守与同组限制；新增科技及宽基重叠、现金变化均为方案组成部分。

## 离线复现

沿主仓环境说明安装锁定依赖，在仓库根目录运行。数据使用`standard/data/2025`，不需要API密钥。

```bash
# 已存账本报表重算，不启动native
PYTHONPATH=src:. python -m scripts.qqq_overlay_report --output runs/qqq-report
# 48月末物理前缀与独立原始数据目标核验
PYTHONPATH=src:. python -m scripts.qqq_overlay_verify --output runs/qqq-causality.json
# 本实验测试需显式指定，根目录默认tests集合保持原样
PYTHONPATH=src:. python -m pytest -q experiments/qqq-overlay/tests --typeguard-packages=scripts

# 真正从规范行情执行Backtrader，先两桥再两科学
PYTHONPATH=src:. python -m scripts.qqq_overlay_run --stage bridge
PYTHONPATH=src:. python -m scripts.qqq_overlay_run --stage science
PYTHONPATH=src:. python -m scripts.qqq_overlay_run
# 对照已发布表逐单元比较，不覆盖不同结果
PYTHONPATH=src:. python -m scripts.qqq_overlay_report --native-output runs/qqq-overlay
```

`run`不传stage默认只验证已生成账户。独立复现任务本地journal位于`evidence/attempts.jsonl`，硬cap8含失败；本次原研究计数另存`evidence/RESEARCH_ATTEMPTS.jsonl`，不得删除已有计数来绕过限制。成功账户只复核，同trial不能换输出路径重跑，失败证据保留。

原Backtrader撮合、现金、分红、拆股、费用和财务核验均复用；新代码仅在`scripts/qqq_overlay_*.py`。科学账户权重直接表达账户目标，部署倍数1，避免40%又被乘95%；100%只是目标，整股、费用和锁定应收仍可造成剩余现金和目标偏差，未启用杠杆。

新参考账本在`reference/OVERLAY/<合约>/`；外层BINDING明确变种，C11只是复用的原生执行壳。五账本、目标、指标和原生标量公开；大ARTIFACT本地留存，可按上面命令重建。本目录单独MANIFEST校验新增材料，原manifest/旧源文件不变。

EXPOSURE中的QQQ＋VGT＋SOXX为这三只基金市值占账户净值的平均值，不是穿透持仓的纯科技行业比例。cash包含尚未可用的分红应收，不重复相加。历史清仓与年度DD口径沿原标准，完整DD不能用年度DD最大值代替。
