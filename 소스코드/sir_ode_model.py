
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import List, Dict, Any

def load_config(config_path: str = 'config.json') -> Dict[str, Any]:
    """설정 파일(config.json)을 불러옵니다."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def sir_differential(t: float, y: List[float], N: int, beta: float, gamma: float) -> List[float]:
    """
    SIR 모델의 미분 방정식 시스템.
    
    Args:
        t (float): 현재 시간 (solve_ivp에 의해 사용됨).
        y (List[float]): [S, I, R] 상태 변수 리스트.
        N (int): 총인구.
        beta (float): 감염 확산 계수.
        gamma (float): 회복률.
        
    Returns:
        List[float]: [dS/dt, dI/dt, dR/dt] 도함수 리스트.
    """
    S, I, R = y
    
    # dS/dt = -β * S * I / N
    dSdt = -beta * S * I / N
    
    # dI/dt = β * S * I / N - γ * I
    dIdt = beta * S * I / N - gamma * I
    
    # dR/dt = γ * I
    dRdt = gamma * I
    
    return [dSdt, dIdt, dRdt]

def run_simulation_and_plot():
    """
    SIR 시뮬레이션을 실행하고 결과를 플로팅합니다.
    """
    # 1. 설정 및 초기 조건 로드
    try:
        config = load_config()
        N = config['POPULATION_SIZE']
        I0 = config['INITIAL_INFECTED']
        S0 = N - I0
        R0 = 0
        beta = config['BETA']
        gamma = config['GAMMA']
        t_span = [0, config['SIMULATION_DAYS']]
        t_eval = np.linspace(t_span[0], t_span[1], t_span[1] + 1)

    except FileNotFoundError:
        print("오류: config.json 파일을 찾을 수 없습니다. 설정 파일을 생성해주세요.")
        return
    except KeyError as e:
        print(f"오류: config.json 파일에 필요한 키({e})가 없습니다.")
        return

    # 2. 미분 방정식 풀이
    # solve_ivp 함수를 사용하여 수치해법으로 SIR 모델을 풉니다.
    solution = solve_ivp(
        fun=sir_differential, 
        t_span=t_span, 
        y0=[S0, I0, R0], 
        args=(N, beta, gamma),
        t_eval=t_eval,
        dense_output=True
    )
    
    S, I, R = solution.y

    # 3. 결과 플로팅
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(solution.t, S, 'b-', label=f'Susceptible (감염 가능자)')
    ax.plot(solution.t, I, 'r-', label=f'Infected (감염자)')
    ax.plot(solution.t, R, 'g-', label=f'Recovered (회복자)')

    # 그래프 디자인 개선
    ax.set_title('SIR Model Differential Equation Simulation', fontsize=16)
    ax.set_xlabel('Days (일)', fontsize=12)
    ax.set_ylabel('Population (인구 수)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # 최대 감염자 수 및 시점 표시
    max_infected_val = np.max(I)
    max_infected_day = solution.t[np.argmax(I)]
    ax.axvline(max_infected_day, color='k', linestyle=':', lw=1.5, 
               label=f'Peak Day: {max_infected_day:.1f}')
    ax.text(max_infected_day + 2, max_infected_val, 
            f'Peak Infected: {int(max_infected_val):,}', 
            fontsize=10, va='center')

    # R0 값 계산 및 표시
    R0_val = beta / gamma
    plt.figtext(0.5, 0.01, rf'$R_0 = \frac{{\beta}}{{\gamma}} = \frac{{{beta:.2f}}}{{{gamma:.2f}}} = {R0_val:.2f}$', 
                ha='center', fontsize=12, bbox={"facecolor":"white", "alpha":0.7, "pad":5})

    plt.tight_layout(rect=[0, 0.05, 1, 1]) # R0 텍스트와 겹치지 않도록 조정

    # 4. 그래프 파일로 저장
    output_filename = 'sir_model_plot.png'
    plt.savefig(output_filename)
    print(f"시뮬레이션 그래프를 '{output_filename}' 파일로 저장했습니다.")
    
    # 그래프 보여주기 (선택 사항)
    # plt.show()

if __name__ == "__main__":
    # Windows에서 Matplotlib 한글 폰트 설정
    try:
        plt.rc('font', family='Malgun Gothic')
        plt.rc('axes', unicode_minus=False)
    except:
        print("경고: 'Malgun Gothic' 폰트를 찾을 수 없어 한글이 깨질 수 있습니다.")
    
    run_simulation_and_plot()
