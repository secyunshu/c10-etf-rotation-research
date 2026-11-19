from datetime import date
from math import isclose

import pytest

from c10standard.io import ROOT
from c10standard.models import Artifact
from rotbt.phase2.models import FeatureStore
from scripts.annual_data import load_prepared
from scripts.v23_rank_weights import allocation, candidates, context, describe, select


def test_real_first_month_order_and_future_price_perturbation_keep_correct_larger_slot() -> None:
    data=load_prepared('ANNUAL')
    pool=next(p for p in data.schedule.entries if p.year==2022)
    day=date(2021,12,31)
    picks=select(candidates(pool,data.features,day))
    assert (picks.first,picks.second)==('XLE','SOXX')
    altered=FeatureStore(frames={s:f.copy() for s,f in data.features.frames.items()})
    for f in altered.frames.values():
        f.loc[f.index>str(day),'tr']*=1000
    assert select(candidates(pool,altered,day))==picks
    assert allocation(picks,'60_30').q=={'XLE':.6,'SOXX':.3,'QQQ':.1}


def test_local_ytd_covers_exact_archived_window_and_original_target_sequence() -> None:
    ctx=context('YTD2026')
    assert ctx.market.last==date(2026,11,11)
    assert len(ctx.picks)==11
    assert (ctx.picks[0].first,ctx.picks[0].second)==('SLV','SMH')
    assert (ctx.picks[-1].first,ctx.picks[-1].second)==('XLE','SMH')
    assert all(d.frame.index[-1].date()==date(2026,11,11) for d in ctx.market.datasets)


def test_descriptive_metrics_use_native_equity_and_reject_fabricated_terminal_profit() -> None:
    artifact=Artifact.model_validate_json((ROOT/
        'experiments/annual-qqq-spy-sleeve/reference/C11/ARTIFACT.json').read_bytes())
    rows=describe(artifact,'HIST','45_45',5)
    assert isclose(rows[-1].stats.cagr,.29241614075663214)
    assert rows[-1].stats.days==1003
    corrupted=artifact.model_copy(deep=True)
    corrupted.account.daily[-1].equity+=100
    with pytest.raises(ValueError):
        describe(corrupted,'HIST','45_45',5)
