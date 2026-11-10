# Fixed one-ETF-per-economic-risk-group validation

Goal: test the user's fixed252-day/monthly rotation hypothesis that limiting correlated economic groups can improve top2/top3 diversification, showing2022–2025 annual return/DD and2026YTD.

Authorization: user explicitly chose original252-day/monthly and fine economic-risk groups on2026-11-09. Existing historicalpool reconstruction is finished separately. Use its conditionalhistorical34riskpool as primary plus paired legacy36 to separate poolchanges fromgroupcap. Preservealloldrules/results/publicv1.0.0. No live/orders/account/purchases/background/publication.

## Frozen data and rules

- Fixedthrough2026-11-06(lastcompleteddate lockedinpriorstudy); don'trollforwardaswallclockadvances. Historicalbacktest2022-01-03–2025-12-31 endopenliquidation; reset2026fromJan2new50k→Nov6closeMTM; continuousfrom2022genesis→Nov6close, no2025endliquidation,2026denominatoritsown2025Dec31close.
- 252-tradingdaygrossTRrank, originaleligibility≥253priorobservations/actualADV20≥$10M/20positivedays/rawclose≥$5; symboltie; monthlyXNYSlastsessionclose→nextopen; prerisk.90/K,SHYresidual,once×.95;50k,tax0,oldfees,5/20bpside,whole-shares/nativefullfillassumptions. No stops/leverage/volbudget/extrafilters.
- Constrainedselection: globalrankgreedy choosefirstcandidatewhosegroupunused, untilKslots orrankexhausted; thenstrictsame252SHYgate; fail/missing fixedslotsreturntoSHY, no newrankaftercashgate/no riskreallocation. UnconstrainedsameKfirstgloballyranked. SHYisdefenderonly, excludedfromriskgroups; defensiveholdingdoesn'tconsumeagroupriskquota.
- Groups21 fixed before newresults. BroadUSstyle includesQQQ/small/dividend/quality; technology+semiconductoronegroup; otherUSsectorsseparate; developedforeignonegroup/emergingforeignonegroup; bondshort/nominalduration/inflation/IG/highyield; metalsGLD+SLV;commodityPDBC. CompleteJSONmapisprotocolinput. Economicgroupingnotempiricaldecorrelation: QQQandVGT may stilloverlapacrossgroups.
- Pools: original36 risk and conditionalhistorical34risk fromlatesthistoryCSV, both+SHY. HistoricalSPMOclasshasnofixedcandidatepasses5B, noMTUM expansioninmainstudy. HistoricalshortwinnerSHYdeduplicatedasdefendernotrisk. Historicalpoolisconditionaloriginal95reportedAUMarchives, notstrictPIT; don'teraseitslimitations.
- Jointinputs45assets=original38container+7newAGG/BIL/HYG/QUAL/SOXX/TIP/VIG. Existingold38economicvaluesunchanged; oldaccountskeep37feedorder, benchmarks2/3feeds, newhistoricalpoolfeeds35ordered. Never feedall45 toeveryaccount.
- New7 historyneed2020-12-31start(2021only252sessions,252returnrequires253prices). ReadonlyFutuquotafirst, K_DAY/NONE/RTH fixedrange2pagesmaximum,rehab,actualcashdetails; max160requestsincludingfailures,one retry/request. No QFQ executions or missingpay guesses. Sourcecompanyactions/splitshareunits strict, includingQUAL/SOXXlate2026split.

## Finite catalog and evidence

- 2pools×2groupconstraints×3K×2costs×3cohorts=72requestedroles. K1capidenticaltosamepoolunconstrained6/pool=12aliases→60uniqueeconomicpaths.
- Legacyunconstrained18fixedpaths already exist (public2022–2025sixreferences, prior2026reset/continuous12artifacts) and mayonlyreusewithmatchinghashes/fees/feed/timing/sourceprefix/contracts. Theseare18reuses, notnewindependentsuccesses. LegacyK1cap6aliasesreusebaselines; historicalK1cap6aliaseshistoricalnewbaseline.
- Newscience42paths:historicalunconstrained18+historicalconstrainedK2/3twelve+legacyconstrainedK2/3twelve. Bridges2fixed:legacyC10K1cost5historicalterminal2025 andlegacyC11K2cost5continuous2026. Plannedactual44. Nativehardcap64includesfixtures/errors/retries; append-onlyseparatejournal andretainedfailureobjects. No borrowingolderbudgets.
- Physicalhistorical≤2025featuresclippedbeforebacktest;2026inputextensionpurposeauthorizednewresearch, oldforwardchain/signals/observationsunchanged.2026hasbeenviewed,allnewresultsareexploratoryconditional,nofreshOOSorwinnerpromotion.

## Execution

- [ ] Freezeplan/protocol/groupmap/poolhashes/catalognamedaliases beforedataAPIandnewstrategyreturns. Independentdesignchallenge verifies mapcoverage44risk+SHY/72roles andcontrolsemantics.
- [ ] Sourcefeasibility/QA first; preserveold38prefixandnew7raw/actionevidence. Missing/specialcash/unresolvedsplit/CIL blocksrelatedpool; don'tdeletefundtounblock. Newissuesneedrootcause/proposedminimumfixbeforeanychangeunderAGENTS.
- [ ] Thinstrategy/nativeorchestration reuse provenBacktraderbroker/fees/companyactionadapter; no custommatching/accountengine. Boundeddecisioncalendar distinctfromMTMend=date.max. PureTDD forgroupuniqueness/greedyrank/SHYslots/top1identity/poolfeedmembership/datecutoff/reusedcontracts.
- [ ] Preparecode/sourcehashlock andbridges. OnlyafterbridgePASS runregistered42science, fill72roleregistry withaliases/reusereceipts. EveryactualCerebrostarted/finished/failedcounted, no repeatedsuccessfulruns.
- [ ] Independentrawsignal,sourcefill/fees/cash/entitlement/split/Position/Trade/NAV/TimeReturn checks; reusedfinancialobjectsvalidated againstfrozenoldbooks. Report2022–25continuousannualreturn/DD plusfullCAGR/vol/Sharpe/cost,2026totalYTD withownopeningNAV,monthly/assetprofit/endholdings; all4pool×capcells inpairedtables.
- [ ] Adversarialresultreview; preservebadoutcomes/no groupredesign/noselecttopKchampion. Save localGit, conciseChinesecomparisons andexplicitremaininguncertainty. No automatictrade/publicupdate.

Expectednewstudy: `research/ETF_GROUP_CAP_20261109/`, inputs `inputs_etf_group_cap_20261109/`, outputs `etf_group_cap_outputs/20261109_v1/`, scripts limited to necessary inputs/targets/run/audit/report adapters. Reuse matureframeworkandexistingutilities; keepcodeproportionaltorequirement.
