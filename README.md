# SIR / SEIRS 미분방정식 기반 감염 확산 시뮬레이터

Python과 SciPy로 감염병 확산을 **연립미분방정식(ODE)** 으로 계산하고, Tkinter/Matplotlib GUI에서 조건을 비교하는 프로젝트입니다. 저장소의 두 실행 경로를 구분해야 합니다.

- `소스코드/sir_ode_model.py`: 기본 **SIR (S/I/R)** 미분방정식 실험
- `소스코드/sir_ode_gui.py`: **S/E/I/R/V/D**를 포함한 확장 구획 모델, 변이·개입 시나리오 및 GUI

## 구현 특징

- SciPy `solve_ivp` 기반 수치 적분
- 잠복기, 자연/백신 면역 지속, 백신 돌파 감염 및 사망 상태
- 특정 날짜 이후의 변이별 파라미터 변경과 개입별 전파율 배수 적용
- 상세 설정과 JSON 시나리오 파일의 파라미터를 결합하는 계산 경로
- Matplotlib 기반 시계열 시각화, Tkinter 기반 설정·시나리오 편집

## 설치 및 실행

Python 3과 GUI(Tkinter)가 설치된 환경에서:

```bash
python -m pip install -r "소스코드/requirements.txt"
cd "소스코드"
python sir_ode_gui.py
```

기본 SIR 모델 실험도 별도 파일로 보관하고 있습니다.

```bash
cd "소스코드"
python sir_ode_model.py
```

**참고:** 기본 SIR 실험 파일은 현재 작업 디렉터리의 `config.json`을 참조하므로 `소스코드`에서 실행해야 합니다.

## 주요 파일

| 파일 | 역할 |
|---|---|
| [`sir_ode_gui.py`](소스코드/sir_ode_gui.py) | 확장 구획 모델과 GUI, 시나리오 편집 |
| [`sir_ode_model.py`](소스코드/sir_ode_model.py) | 기본 SIR 미분방정식 계산 |
| [`virus_presets.json`](소스코드/virus_presets.json) | 기본 바이러스 파라미터 예시 |
| [`variants.json`](소스코드/variants.json) | 시간별 변이 시나리오 |
| [`interventions.json`](소스코드/interventions.json) | 날짜별 전파 억제 시나리오 |
| [`development_log.md`](소스코드/development_log.md) | 변경 이력 및 과거 오류 기록 |

## 확인되지 않은 사항과 한계

기존 개발 로그에는 GUI 편집기, UI 렌더링, `solve_ivp` 호출 관련 오류가 기록돼 있습니다. 뒤이은 수정 기록도 있으나, **현재 코드의 실제 GUI 실행과 모든 시나리오 동작을 재검증한 결과는 아닙니다.** 데스크톱 실행 환경·폰트·데이터 파일 경로에 따라 다를 수 있습니다.

파라미터와 시나리오는 교육 및 수치 모델 실험용이며, 실제 감염 규모 예측이나 의료·정책 판단에 사용하기 위한 검증 모델이 아닙니다.

## 시리즈

- [sir-model-classic](https://github.com/Transient-Onlooker/sir-model-classic): 격자 기반 확률적 SIRVQ-D 모형
- [sir-model-diff-web](https://github.com/Transient-Onlooker/sir-model-diff-web): TypeScript/RK4 기반 웹 시뮬레이터
