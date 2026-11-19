# 本研究约束

遵循 /Users/yunshu/.claude/python-security-constraints.md；函数数据传输使用Pydantic DTO或既有框架模型，beartype验证，测试typeguard；新Python文件严格mypy+Pyright及Ruff。已有Model映射字段不构成新的裸dict函数接口。源文件限scripts/v23_rank_weights.py，测试限本目录，避免改冻结引擎/标准/旧输入。Backtrader负责订单、现金、费用、持仓、公司行动和收益，研究仅产生目标并独立核验。纯本地，不使用QC、API或交易接口。
