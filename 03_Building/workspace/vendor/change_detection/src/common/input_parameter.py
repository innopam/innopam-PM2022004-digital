# 정제할 유형 목록
refine_types = {
    1: "정상",
    2: "육안확인불가",
    3: "그림자(일부)",
    4: "나무(일부)",
    5: "옥외주차장",
    6: "옥상정원",
    7: "부속건물",
    8: "잘린건물",
    9: "기타"
}


def show_refine_types():
    """
    정제할 유형을 사용자에게 보여줍니다.
    """
    print("<정제할 유형 목록>")
    for key, value in refine_types.items():
        print(f"{key}: {value}")
    print("\n")


def get_multiple_inputs_with_defaults():
    """
    사용자로부터 5개의 파라미터를 한 번에 입력받고, 입력이 없을 경우 기본값을 설정합니다.
    """
    # 기본값을 사용할지 물어봅니다.
    use_defaults = input("기본값을 사용하려면 Enter를 누르세요. (직접 입력하려면 'n' 입력): ").strip().lower()

    if use_defaults == '':
        # 기본값 적용
        return 0, 50, [2, 8], 0.6, [0, 100], [0, 100]
    else:
        # 최소 면적 및 최대 면적 입력
        min_area = int(input("최소 면적을 입력하세요 (기본값: 0): ") or 0)
        max_area = int(input("최대 면적을 입력하세요 (기본값: 50): ") or 50)

        # 정제할 유형 목록을 보여준 다음에 사용자 입력을 받음
        show_refine_types()
        refine_categories = input("정제할 유형을 쉼표로 구분하여 입력하세요 (기본값: 2,8): ") or "2,8"
        refine_categories = [int(x.strip()) for x in refine_categories.split(',')]

        # 입력한 정제 유형이 올바른지 확인 (유효한 값만 허용)
        refine_categories = [category for category in refine_categories if category in refine_types]
        if not refine_categories:
            refine_categories = [2, 8]  # 유효한 입력이 없을 경우 기본값 설정

        # 탐지 기준과 면적 범위 입력
        detection_threshold = float(input("건물 추론 탐지 기준을 입력하세요 (기본값: 0.6): ") or 0.6)

        # 건물 추론 평가 면적 범위 입력 및 기본값 처리
        detection_area = input("건물 추론 평가 면적별 분석 범위를 입력하세요 (기본값: 0,100): ") or "0,100"
        detection_area = [int(x.strip()) for x in detection_area.split(',')]

        # 건물 변화 탐지 평가 면적 범위 입력 및 기본값 처리
        change_detection_area_range = input("건물 변화 탐지 평가 면적별 분석 범위를 입력하세요 (기본값: 0,100): ") or "0,100"
        change_detection_area_range = [int(x.strip()) for x in change_detection_area_range.split(',')]

        return min_area, max_area, refine_categories, detection_threshold, detection_area, change_detection_area_range

