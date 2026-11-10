# ETF13612W固定加权评分对照

> For agentic workers: use subagent-driven-development for isolated implementation and independent review. No native account in tests or agents; root executes the finite catalog.

Goal: answer the user's request for annual return and maximum drawdown of calendar-month13612W top1/top2/top3 on the standard historical pool with risk grouping, versus SPY/QQQ and the unchanged252-day rule.

Authorization: user explicitly requested the experiment and selected real month-end lookbacks on2026-11-10. This is a new research variant, not permission to replace V2/defaultK1, tune weights, annual-refresh the universe or trade.

## Frozen hypothesis

- Calendar-month ends: for each decision day t (last XNYS session of its month), Rm=TR(t)/TR(last XNYS session of calendar month t−m)−1 for m=1,3,6,12. Require exact endpoints, never pad missing data or substitute21/63/126/252. Initial historical signal2021-12-31 has2020-12-31 prior12month endpoint. Keep minimum253 observations from V2.
- Score=(12R1+4R3+2R6+R12)/4. Dividing by4 is a common scale, not normalized weights that sum to1, and does not change rank or score>SHY comparisons. SHY uses the same score; no added score>0/SMA/canary filter. This only tests a scoring component, not a full named FAA/PAA/VAA system.
- Pool remains V2's conditional pre2022 archived-AUM historical34 risk names＋SHY and21 fixed economic groups. Global score descending, symbol ascending; greedy skip occupied groups untilK slots; then strictscore>SHYscore; failed/missing slots go toSHY, no backfill/reweight. Original rawclose≥5/ADV20≥10M/20positive-volume observations, .9/K slots then once×.95 deployment,50k,whole shares,tax0,originalfees,5/20bp per side.
- Monthly close→nextopen with original native fullfill assumptions. Historical2022-01-03–2025-12-31 last-dayopenliquidation; reset2026 Jan2–Nov6 closeMTM; continuous2022genesis→Nov6 with no2025 liquidation, own2025Dec31close NAV denominator. All2026 is already seen; no freshOOS claims or annualizedYTD performance headline.
- Main evidence is2022–2025 annual/total/CAGR/DD and matched252. Show2026 separately and retain both starting states. Show fees/turnover/orders/volatility/Sharpe and20bp sensitivity. No optimization or result-dependent testing of other window/weight/frequency/group choices.

## Finite evidence and minimal code

- New science:3K×2costs×3cohorts=18native accounts. Two fixed bridges use original252 score on the same historicalpool/groupcap:K1/5bp historical;K2/5bp continuous. Planned20starts; hard28includingfailures/tests/retries, separateappend-onlyjournal. Never borrow or reset old budgets. MarketAPI0.
- Reuse36publicV2 accounts:3original252 rules+SPY/QQQ/6040 ×2cost×3cohort, each with exact source/hash/time/settings identity. Report54roles total:18new+36reused. Bridges2are implementation validation, not new independent strategy successes.
- Reuse scripts.etf_c10_2026_run._native/audit/export_books and accepted groupinput loaders/financial checks through scoped target+oracle hooks restored finally. Backtrader owns orders/broker/cash/Position/fees/TimeReturn. No new account engine or copy of core; outer trial metadata identifies13612W while old C10/C11/C00 shell is only K.
- New signal implements real calendar ends; independent oracle derives rawTR endpoints and liquidity separately. Neither reads future rows. History input uses physical2025 data before constructing features;2026 is a separately authorized read. Keep source/config/publicV2/core files byteimmutable and bind them before native.

## Tasks and acceptance

- [ ] Record/freeze this plan, protocol/formula/catalog and exact pool/group source hashes before implementation/native; independent challenge resolves endpoint/gate/cohort/reuse assumptions.
- [ ] PureTDD: known-score arithmetic, weekend/holiday monthends, months with unequal session counts, exact12monthendpoint, missing/NaNfail, future rows irrelevant, symbolties/groupquota/SHYfixedslots, hookrestore, hard28includingfailures/no silent reruns.
- [ ] Minimal targets and runner adapters, source lock, Pydantic boundaries/typeguard/strictmypy/Pyright/Ruff; targeted existing input regression, no full native fixtures. Independent spec+code review beforefirstCerebro.
- [ ] Root2bridges exact toV2 books/targets/native return, then18registered accounts once. Save rawARTIFACT andcompletionmarker beforepostrunvalidation; preserve failure evidence and resume completed native objects without new account runs.
- [ ] Independently validate monthly endpoints/weighted targets and financial ledger; generate complete54role common-window annual/total/YTD/cost tables and comparisonplot. Explicitly distinguish native20starts from18science and36reuses.
- [ ] Challenge findings: more recent information may improve reversals or cause whipsaw; inspect matched return/DD/turnover, coststress and annual concentration. No universalvalid/invalid conclusion from seen samples. Preserve all outcomes and currentstandard. Save localGit; research artifact publication, if done, must remain labeled exploratory rather than modifying standard rules.

Files: research/ETF13612W_20261110/{PROTOCOL,CATALOG,RUN_LOCK,FINAL_REPORT,...}, etf13612w_outputs/20261110_v1/, scripts/etf_13612w.py and focusedtests; use only additional smallreport adapter if existing statistics cannot be called directly.
