from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from typeguard import typechecked

from c10standard.io import ROOT
from scripts.current_standard_ranked import load_policy, select_target
from scripts.ytd_annual_data import load_prepared


@typechecked
def test_current_machine_pointer_selects_authorized_ranked_60_30_and_keeps_sleeve() -> None:
    policy=load_policy(ROOT)
    assert policy.policy_version=='2.3.1'
    assert policy.ranked_slot_weights==(.6,.3)
    assert policy.sleeve_weight==.1 and policy.shy_momentum_gate is False


@pytest.mark.parametrize(('first','sleeve'), [('QQQ','SHY'),('XLE','QQQ')])
def test_actual_default_selector_uses_ranked_weights_without_shy_gate(first,sleeve):
    market=load_prepared('ANNUAL','science')
    day=date(2026,10,30)
    for symbol,frame in market.features.frames.items():
        frame.loc[pd.Timestamp(day),'mom252']={first:1.,'SMH':.5,'SHY':100.}.get(symbol,-.2)
    target=select_target(market.schedule.entries[-1],market.features,day)
    assert target.q=={first:.6,'SMH':.3,sleeve:.1}
    for symbol,frame in market.features.frames.items():
        frame.loc[pd.Timestamp(day),'mom252']={'SMH':1.,first:.5,'SHY':100.}.get(symbol,-.2)
    reversed_target=select_target(market.schedule.entries[-1],market.features,day)
    assert reversed_target.q=={'SMH':.6,first:.3,sleeve:.1}


def test_changed_strategy_is_rejected_by_pointer_hash(tmp_path:Path):
    folder=tmp_path/'standard/v2_3_1';folder.mkdir(parents=True)
    (tmp_path/'standard/CURRENT.json').write_bytes((ROOT/'standard/CURRENT.json').read_bytes())
    text=(ROOT/'standard/v2_3_1/STRATEGY.json').read_text()
    (folder/'STRATEGY.json').write_text(text.replace('0.6','0.5'))
    with pytest.raises(ValueError,match='hash'):
        load_policy(tmp_path)


def test_policy_model_rejects_equal_weights_even_with_updated_hash():
    from scripts.current_standard_ranked import Policy
    policy=load_policy(ROOT)
    values=policy.model_dump()
    values['ranked_slot_weights']=(.45,.45)
    with pytest.raises(ValueError):
        Policy.model_validate(values)


def test_frozen_equal_weight_selector_and_previous_machine_file_remain_reproducible():
    from scripts.current_standard import Policy as PriorPolicy
    from scripts.current_standard import select_target as prior_select
    prior=PriorPolicy.model_validate_json((ROOT/'standard/v2_3/STRATEGY.json').read_bytes())
    assert prior.risk_slot_weight==.45
    market=load_prepared('ANNUAL','science')
    day=date(2026,10,30)
    old=prior_select(market.schedule.entries[-1],market.features,day)
    new=select_target(market.schedule.entries[-1],market.features,day)
    assert old.q=={'XLE':.45,'SMH':.45,'QQQ':.1}
    assert new.q=={'XLE':.6,'SMH':.3,'QQQ':.1}
