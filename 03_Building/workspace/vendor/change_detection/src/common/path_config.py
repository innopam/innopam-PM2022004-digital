import os
import json

regions = [
    "gangseo", "junggu", "jungrang", "mapo", "seocho",
    "songpa", "suseo", "yangcheon", "youngdeungpo"
]

# 마지막 선택 값을 저장할 파일 경로
last_selection_file = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../config/last_selection.json')
)

# 이전 선택값을 불러오는 함수
def load_last_selection():
    """
    이전에 선택한 지역, 연도, 이전 연도를 저장한 파일에서 불러옴.
    """
    if os.path.exists(last_selection_file):
        with open(last_selection_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# 마지막 선택값을 저장하는 함수
def save_last_selection(region, year, previous_year):
    """
    선택한 지역, 연도, 이전 연도를 파일에 저장.
    """
    with open(last_selection_file, 'w', encoding='utf-8') as f:
        json.dump({"region": region, "year": year, "previous_year": previous_year}, f)


# JSON에서 경로를 로드하고 동적으로 {region}, {year}, {previous_year}을 대체
def load_paths(region, year, previous_year):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # DT 폴더 기준

    config_path = os.path.join(base_dir, "config", "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    paths = config["building"]

    for key, path in paths.items():
        full_path = os.path.join(base_dir, path.format(year=year, previous_year=previous_year, region=region))
        paths[key] = full_path

    return paths


# 지역을 선택하는 함수 (이전 값이 있으면 기본값으로 사용)
def select_region():
    """
    사용자가 지역을 선택하도록 하고, 이전에 선택한 값이 있으면 기본값으로 사용.
    """
    last_selection = load_last_selection()
    default_region = last_selection["region"] if last_selection else None

    print("지역을 선택하세요:")
    print("0: all")  # all 옵션 추가
    for i, region in enumerate(regions):
        print(f"{i + 1}: {region}")

    prompt = f"선택한 지역의 인덱스를 입력하세요 (0-{len(regions)}, 기본값: {default_region}): " if default_region else f"선택한 지역의 인덱스를 입력하세요 (0-{len(regions)}): "
    selected_index = input(prompt).strip()

    if selected_index == '' and default_region:
        return default_region
    elif selected_index == '0':
        return "all"
    elif selected_index.isdigit() and 1 <= int(selected_index) <= len(regions):
        return regions[int(selected_index) - 1]
    else:
        print("유효하지 않은 선택입니다. 다시 시도하세요.")
        return select_region()

# 연도를 선택하는 함수 (이전 값이 있으면 기본값으로 사용)
def select_year(prompt, last_selection, key):
    """
    연도 선택 시, 이전에 선택한 값을 기본값으로 사용.
    """
    default_year = last_selection[key] if last_selection else None
    year_input = input(f"{prompt} (기본값: {default_year}): ").strip()
    return year_input if year_input else default_year
