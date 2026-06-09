import os
import json


def load_building_paths(region, year, previous_year):
    # DT 디렉토리 기준 절대경로 계산
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # config 경로도 base_dir 기준으로
    config_path = os.path.join(base_dir, 'config', 'config.json')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    paths = config['building']

    # 치환 + 절대경로 변환
    for key, path in paths.items():
        if not key.startswith('_'):
            rel_path = path.format(region=region, year=year, previous_year=previous_year)
            paths[key] = os.path.normpath(os.path.join(base_dir, rel_path))

    return paths
