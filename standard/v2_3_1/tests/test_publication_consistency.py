import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_repository_landing_title_and_current_pages_name_the_machine_version():
    current = json.loads((ROOT / 'standard/CURRENT.json').read_text())
    version = current['policy_version']
    readme = (ROOT / 'README.md').read_text()
    assert readme.splitlines()[0] == f'# 美股 ETF 动量轮动 · V{version}'
    for name in ('docs/CURRENT_STANDARD.md', 'docs/REPRODUCE_V2.md',
                 'standard/v2_3_1/REPORT.md'):
        title = (ROOT / name).read_text().splitlines()[0]
        assert f'V{version}' in title, name
    assert f'python -m {current["entrypoint"]}' in readme
    assert current['strategy_path'] in readme


def test_published_current_documents_do_not_claim_the_release_is_local_only():
    for name in ('README.md', 'standard/v2_3_1/REPORT.md'):
        text = (ROOT / name).read_text()
        assert '本次仅本地保存' not in text, name
        assert '未创建远端发布' not in text, name
        assert '不推送GitHub' not in text, name
        assert 'https://github.com/secyunshu/c10-etf-rotation-research/releases/tag/v2.3.1' in text
