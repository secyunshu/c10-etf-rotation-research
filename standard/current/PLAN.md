# V2.2年度池成为当前默认

用户2026-11-11明确要求将每年换池加入默认并推送GitHub。本任务采用已有D完整年度池政策，不改变任何选择/费用参数、不修改历史资料、不将B仅换SMH诊断升格。

- 当前配置新建standard/CURRENT.json及standard/current/STRATEGY.json，policy_version=2.2.0、top_k=2、universe_refresh=annual、月度交易/252日/21风险组/SHY门槛/原95%部署保持。冻结年度schedule直接绑定experiments/v2-annual-repool/ANNUAL.json和已验证hash。
- 新默认入口scripts/current_standard.py从CURRENT解析政策，默认只读验证并重算参考报告；显式--execute才通过既有annual_run先2fixed桥再2annual成本账户，完整复用Backtrader/财务审计。完成后对参考五账/目标/timing/原生日收益/指标逐项核对，不仅比较收益。
- 本次发布独立native计划4/hard8含失败，用新的固定release journal；scope只替换annual_run.journal_path/journal_event，退出finally恢复，不使用旧4/8计数、不修改任何旧源码。默认入口必须实际运行，不靠README宣传替代。重复同路径仅复核、失败留证/不换路径重复同key。
- 原standard/STRATEGY.json、src/c10standard及全部旧实验源、报告、manifest原样保留作为旧固定池复现；README、当前标准说明和复现文档明确新默认及旧入口的兼容身份。所有旧冻结源锁仍可按原逻辑核验。固定池2026收益保历史对照，不写成年度池2026收益。
- 新默认支持已验证2022–2025年度池；没有2026年度池验收或未来选基工具。本任务不静默将2025池延伸到2026、不新增历史数据/前瞻/账户/交易接口。未来年度名单需要按当时可得资料建立、记录证据和验证，不能靠当前指针假定已存在。
- 保留归档资料异步/最长陈旧220或225日、INDA更新来源反例、SMH贡献集中及暂无普适年度alpha证据。用户采用是政策选择，不是新的OOS通过，不自动修正旧池或优化更多参数。
- 完成标准：当前配置/文档/入口/本地ETF_STANDARD_CURRENT指针一致，TDD默认解析/防篡改/年度身份/预算/复用验证和safe回归，严格mypy/Pyright/Ruff，4native与现存reference匹配、source locks保持，独立review后GitHub合并+v2.2.0标签+隔离下载hash/tests/报告核对。发布所有必要代码及说明，保持旧版本可追溯。
