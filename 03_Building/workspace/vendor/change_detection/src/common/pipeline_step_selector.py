def get_selected_pipeline_indices(pipeline_steps):
    """
    사용자로부터 실행할 pipeline 인덱스를 받아 리스트로 리턴
    """
    print("실행하고자 하는 단계를 선택하십시오 (인덱스를 쉼표로 구분하여 여러 개 선택 가능):")
    for i, (description, _) in enumerate(pipeline_steps):
        print(f"{i + 1}: {description}")

    selected = input("선택한 단계의 인덱스를 입력하십시오 (Enter를 누르면 모든 단계 실행): ").strip()

    if selected == '':
        return list(range(len(pipeline_steps)))  # 전체 실행
    else:
        return [int(i.strip()) - 1 for i in selected.split(',') if i.strip().isdigit()]


def run_selected_pipeline_steps(pipeline_steps, selected_indices):
    """
    인덱스 리스트에 따라 지정된 단계 실행
    """
    for idx in selected_indices:
        if 0 <= idx < len(pipeline_steps):
            _, func = pipeline_steps[idx]
            func()
        else:
            print(f"{idx + 1}번 인덱스는 유효하지 않습니다.")