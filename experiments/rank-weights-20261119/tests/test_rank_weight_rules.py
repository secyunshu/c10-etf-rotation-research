from datetime import date

import pytest
from typeguard import typechecked

from scripts.v23_rank_weights import Candidate, Candidates, OrderedPicks, allocation, select


@typechecked
def test_larger_weight_follows_rank_not_symbol_and_names_flip_rebalances() -> None:
    first = OrderedPicks(day=date(2026,1,30),first='SLV',second='SMH')
    assert allocation(first,'60_30').q=={'SLV':.6,'SMH':.3,'QQQ':.1}
    flipped = first.model_copy(update={'first':'SMH','second':'SLV'})
    assert allocation(flipped,'60_30').q=={'SMH':.6,'SLV':.3,'QQQ':.1}
    assert allocation(first,'45_45').q==allocation(flipped,'45_45').q


def test_all_three_weights_keep_total_and_original_conditional_sleeve() -> None:
    for name,pair in [('45_45',(.45,.45)),('50_40',(.5,.4)),('60_30',(.6,.3))]:
        picks = OrderedPicks(day=date(2025,1,31),first='XLE',second='SPY')
        q = allocation(picks,name).q
        assert q=={'XLE':pair[0],'SPY':pair[1],'SHY':.1}
        assert sum(q.values())==pytest.approx(1.)
    with pytest.raises(ValueError):
        OrderedPicks(day=date(2025,1,31),first='SHY',second='GLD')
    with pytest.raises(ValueError):
        OrderedPicks(day=date(2025,1,31),first='GLD',second='GLD')


def test_ties_group_caps_and_qualification_do_not_change_when_weights_change() -> None:
    values=Candidates(day=date(2025,1,31),rows=[
        Candidate(symbol='VGT',group='tech',score=.3),
        Candidate(symbol='SMH',group='tech',score=.3),
        Candidate(symbol='SLV',group='metal',score=.2),
        Candidate(symbol='XLE',group='energy',score=.4,qualified=False)])
    picks=select(values)
    assert (picks.first,picks.second)==('SMH','SLV')
    assert allocation(picks,'50_40').q=={'SMH':.5,'SLV':.4,'QQQ':.1}


def test_insufficient_distinct_groups_fails_instead_of_filling_with_defender() -> None:
    values=Candidates(day=date(2025,1,31),rows=[
        Candidate(symbol='VGT',group='tech',score=.3),
        Candidate(symbol='SMH',group='tech',score=.2)])
    with pytest.raises(ValueError):
        select(values)
