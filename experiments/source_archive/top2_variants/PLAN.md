# V2 top2 finite momentum-variant study

> For agentic workers: use subagent-driven-development for isolated signal/run implementation and independent source/result review. Root alone runs registered native accounts; tests/agents may not invokeCerebro.

Goal: continue the user-authorized search with six economically interpretable changes around V2 top2, avoiding uncontrolled combinations. User prioritizes return but has discussed top2 as a balanced research candidate; this is a research anchor, not a change of published defaultK1 or investment approval.

## Prior coverage and finite design

Old C06(skip21), C07(rank average63/126/252), C08(reg252), C09(R252/vol63), C21(SMA200gate) were tested on2026-selected legacy36 primarily top3 and2022–2024 development. They cannot substitute for exact historical34/groupcap/top2 comparisons. This round is a conditional replication/transfer for those ideas, not six independent discoveries. Calendar13612W already completed and becomes a frozen top2 comparison, never retuned.

Use acceptedV2 historical34risk+SHY,21groups onepergroup, monthly lastXNYSclose→nextopen, min253 observations, rawclose≥5, actualADV20≥10M and20positive-volume days. Keep fixed2slots, each.45 predeployment; failedslotsSHY, wholeallocation×.95 once,50k,originalfees,tax0,whole shares,5/20bp. No inverse-vol position weights, statefulrankbuffer, stops, leverage, newassets, annualpoolupdates, frequency search or parameter grid.

Six pre-specified variants (alltop2):

|ID|Single mechanism|Ranking|Defense after greedy group selection|
|---|---|---|---|
|E01|Skip latest21tradingdays|TR(t−21)/TR(t−252)−1|same skip score strictly>SHYskipscore|
|E02|Equal-weight four calendar windows|(R1+R3+R6+R12)/4, exactlastXNYSsession of t−m calendar month|same equal score strictly>SHYequal score|
|E03|Risk-adjusted ranking|R252/max(annualizedsamplevol63,0.01)|original unscaledR252 strictly>SHYR252|
|E04|Smoothed regression ranking|252×OLS slope(logTR last252prices)×R²|originalR252 strictly>SHYR252|
|E05|Positive-return defense|originalR252|R252>max(SHYR252,0)|
|E06|Long-trend defense|originalR252|R252>SHYR252 and TR(t)>mean(last200TR)|

For every variant sortdescscore/symbolasc, greedy skip usedgroups until2slots, THEN apply defense to each slot. No replacement aftergate/no riskreallocation. E03/E04 deliberately retain originalSHYgate to isolate rank rather than also changinggate. Volfloor1percent annualized from existing C09 definition, not tuned; zero volatility allowed usingfloor. E04 uses last252 pricepoints whileR252 needs253, matching establishedreg252. E01 is trading-day12−1 approximation, explicitlynotcalendar12−1. E02 uses realcalendarends, like prior13612W; differences against252 includehorizonbasis, against13612W isolateweights.

## Dates, roles, budget and controls

- Main:2022-01-03–2025-12-31, lastdayopenliquidation, continuous annuals. Reset2026Jan2→Nov6closeMTM; continuous2022→Nov6 without2025liquidation, own2025Dec31closeNAV denominator. 2026alreadyseen, no freshOOS. Prices physicallybounded2025 beforefeatureconstruction.
-36newscience=6variants×2cost×3cohorts. Twofixedoriginal252top2/5bp bridges: historicalterminal andcontinuous2026. Planned38actualstarts, hard48includingfailures/tests/retries, separateappend-onlyjournal. NoAPI/native outsidecatalog. Failures preserved, savedcompleteartifacts may revalidate without rerun.
-30reusedroles:original252top2 + SPYBH + QQQBH + SPY60IEF40 =24publicV2 roles, plus prior13612Wtop2 sixroles.66reportedroles total. No newnative forreuses, notindependenttrials. Do not rerun top1/top3 fornewvariants; thislimitsbreadth to mechanisms atfixedK.
- Nativeexecution remains scripts.etf_c10_2026_run._native/audit/export_books + existingcash/splitadapter. Signals andindependentraworacle onlycustom; no matching/accountengine. Outertrialidentity distinctfromoldC11shell. Alloriginalcodes/data/publicV2/oldoutputs immutable andhashbound.

## Pre-result evaluation

- Reportall66 roles; main2022–25 total/CAGR/DD,yearlyreturns/DD,Sharpe rf0/volatility/fees/orders/turnover andcoststress. Matchedrelative differences tooriginal252K2 andcalendar13612WK2; do notmixannualizedYTD withCAGR.
- HistoricalPareto flag only if CAGR≥original andDD≤original, at leastone strictimprovement, atboth5/20costs. Thisisdescriptive,notageneralPASSorpermissiontolive. Reportnumberofyearsbeatingbaseline andworseningworstyear, no hidea failedyearbehindfullCAGR.2026bothstartingstates separate,notselectivegate to rescuehistoryfailure.
- Reuse mature `rotbt.statistics.max_bootstrap` on daily netreturn differences vsoriginal252top2 for the complete1003-session2022–25 window. Fixedfamilyall6includingidenticalpaths, B10000,blocks5/20/60, existingseed20261105+block, bothcosts. Reporteachadjustedone-sidedp andpointwise95CI forannualizedarithmeticmean excess, NOTCAGR CI. Local6multiplicity only; doesnotcorrectearlierhundredsoftrials orcreatefreshOOS. Onlyclaim localstatisticalsupport ifthatvariantadjustedp<.05acrossall3blocks/bothcosts; otherwiseinconclusive, regardlessfamily p. Do not optimize blocklength.
- Same-pathhashes disclose identicaloraliasedrealizedsignals/NAV asoneeconomicpath,notrepeatindependentwins.
- Data causality beforeinterpretingresults: atregisteredmonthlycloses, recompute usedfeatures fromrawprefix only andcompare withfullfeatures; separate raworacle forscore/gate/rank. Check every regularorderdate equalsnextsessionofregisteredtarget. ThisisnotanextraD−1 performancecomparison; the latter remains a separateexecution-sensitivityidea.

## Implementation and acceptance tasks

- [ ] Freezeplan/protocol/formulas/catalogandold-coverage inventory, independentplanchallenge. No returns examinedbeforefreeze.
- [ ] Thin newsignals/run adapter andpureTDD knownscore/regression/volfloor/skipendpoints/calendar/strictgates/fixedslots/no lookahead/hooksrestore/journal48/partialfailclosed. Publicboundariestyped,beartype/typeguard,strictmypy/Pyright/Ruff. No extra brokerfixtures.
- [ ] Reportadapter reuses savedNAVstatistics andlocalbootstrap, verifiesregistry66 andfullboundhashes, emitsallannual/mainYTD/monthly/paired/statistical tables. Validate statistical family/dates independently, test iffeasible usingpurearrays noaccount.
- [ ] Independentimplementationreview; rootsourcefreeze;2bridgesexact then36science. Preserve failures/dataissues; noapprove-newdatacorrectionassumed.
- [ ] Independent rawsignal/monthend/finance/NAV/period/CI resultreview andadversarial narrative. Reportallnegatives, classifywhichmechanism deservesfutureobservation withoutreplacingstandard. SaveGit andseparatelylabeledresearchsnapshot onGitHub perworkflow, no oldstandard update.

Paths: research/ETF_MOMENTUM_VARIANTS_20261110/, etf_momentum_variants_outputs/20261110_v1/, minimal scripts/etf_momentum_variants.py and scripts/etf_momentum_variants_report.py withfocusedtests. Reuseacceptedinputs andframework; allnewmarketAPI0.
