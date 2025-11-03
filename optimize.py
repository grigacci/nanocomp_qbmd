"""
Otimização multi-objetivo de detectores QBMD usando NSGA-II.
"""

import numpy as np
import subprocess
from pathlib import Path
from datetime import datetime

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

from config import loadConfig
from extract import getSimulationResult
from models import SimulationResult


class QBMDOptimizationProblem(Problem):
    """
    Problema de otimização multi-objetivo para detectores QBMD.
    
    Variáveis de decisão (7 parâmetros do simulador):
    - ARG1: Número de períodos da primeira estrutura (int)
    - ARG2: Parâmetro material 1 (float)
    - ARG3: Parâmetro material 2 (float)
    - ARG4: Espessura/composição (float)
    - ARG5: Número de períodos da segunda estrutura (int)
    - ARG6: Parâmetro material 3 (float)
    - ARG7: Parâmetro material 4 (float)
    
    Objetivos (maximizar):
    1. Energia do pico principal
    2. Intensidade da fotocorrente
    3. Fator de qualidade
    4. Proeminência do pico
    """
    
    def __init__(self, config, bounds):
        """
        Parâmetros
        ----------
        config : dict
            Configuração carregada do TOML
        bounds : dict
            Limites inferior e superior para cada variável
            Formato: {'lower': [x1_min, x2_min, ...], 'upper': [x1_max, x2_max, ...]}
        """
        self.config = config
        self.simulation_counter = 0
        self.base_output_dir = Path("./optimization_runs")
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Definir limites das variáveis
        xl = np.array(bounds['lower'])
        xu = np.array(bounds['upper'])
        
        # 7 variáveis, 4 objetivos, 0 restrições
        super().__init__(
            n_var=7,
            n_obj=4,
            n_ieq_constr=0,
            xl=xl,
            xu=xu
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Avalia a população de soluções.
        
        Parâmetros
        ----------
        X : array (pop_size, 7)
            Matriz com os indivíduos da população
        out : dict
            Dicionário para armazenar os objetivos
        """
        F = []  # Matriz de objetivos
        
        for idx, individual in enumerate(X):
            # Executar simulação
            result = self._run_simulation(individual)
            
            # NSGA-II minimiza, então negamos para maximizar
            objectives = [
                -result.max_energy,          # Maximizar energia do pico
                -result.max_photocurrent,    # Maximizar intensidade
                -result.quality_factor,      # Maximizar fator de qualidade
                -result.prominence           # Maximizar proeminência
            ]
            
            F.append(objectives)
            
            self.simulation_counter += 1
            
            # Log de progresso
            if self.simulation_counter % 5 == 0:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Simulação {self.simulation_counter}")
                print(f"  Parâmetros: {individual}")
                print(f"  Energia: {result.max_energy:.2f} eV")
                print(f"  Intensidade: {result.max_photocurrent:.6e}")
                print(f"  Q: {result.quality_factor:.2f}")
                print(f"  Proeminência: {result.prominence:.4f}")
        
        out["F"] = np.array(F)
    
    def _run_simulation(self, parameters):
        """
        Executa o simulador QBMD com os parâmetros fornecidos.
        
        Parâmetros
        ----------
        parameters : array [ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, ARG7]
        
        Retorna
        -------
        SimulationResult
        """
        # Converter parâmetros para formato correto
        arg1 = int(parameters[0])  # Inteiro
        arg2 = f"{parameters[1]:.1f}d0"  # Formato Fortran
        arg3 = f"{parameters[2]:.1f}d0"
        arg4 = f"{parameters[3]:.1f}d0"
        arg5 = int(parameters[4])  # Inteiro
        arg6 = f"{parameters[5]:.1f}d0"
        arg7 = f"{parameters[6]:.1f}d0"
        
        # Diretório de saída específico para esta simulação
        output_dir = self.base_output_dir / f"sim_{self.simulation_counter:05d}"
        
        # Executar script run.sh
        script_path = Path(self.config["simulator"]["exec_path"])
        
        try:
            result = subprocess.run(
                [
                    str(script_path),
                    str(arg1), arg2, arg3, arg4,
                    str(arg5), arg6, arg7,
                    str(output_dir)
                ],
                capture_output=True,
                text=True,
                timeout=self.config["simulator"]["timeout"]
            )
            
            if result.returncode != 0:
                print(f" Erro no simulador (sim {self.simulation_counter}):")
                print(result.stderr)
                return self._default_result()
            
            # Extrair métricas do resultado
            photocurrent_file = output_dir / self.config["paths"]["photocurrent"]
            
            if not photocurrent_file.exists():
                print(f"  Arquivo não encontrado: {photocurrent_file}")
                return self._default_result()
            
            simulation_result = getSimulationResult(
                str(photocurrent_file),
                self.config["analysis"]["peak_threshold"]
            )
            
            return simulation_result
            
        except subprocess.TimeoutExpired:
            print(f"  Timeout na simulação {self.simulation_counter}")
            return self._default_result()
        except Exception as e:
            print(f"  Erro ao executar simulação {self.simulation_counter}: {e}")
            return self._default_result()
    
    def _default_result(self):
        """Retorna resultado padrão para simulações falhas"""
        from models import SimulationResult
        return SimulationResult(
            max_energy=0.0,
            max_photocurrent=0.0,
            prominence=0.0,
            quality_factor=0.0
        )


def run_optimization():
    """Executa a otimização multi-objetivo com NSGA-II"""
    
    # Carregar configuração
    config = loadConfig("config.toml")
    
    # Definir limites das variáveis
    # AJUSTE ESTES VALORES DE ACORDO COM SEU PROBLEMA!
    bounds = {
        'lower': [
            1,      # ARG1: mínimo 1 período
            1.0,    # ARG2: mínimo valor do parâmetro
            5.0,    # ARG3: mínimo valor do parâmetro
            1.0,    # ARG4: mínimo espessura
            1,      # ARG5: mínimo 1 período
            1.0,    # ARG6: mínimo valor do parâmetro
            3.0     # ARG7: mínimo valor do parâmetro
        ],
        'upper': [
            10,     # ARG1: máximo 10 períodos
            5.0,    # ARG2: máximo valor do parâmetro
            10.0,   # ARG3: máximo valor do parâmetro
            5.0,    # ARG4: máximo espessura
            10,     # ARG5: máximo 10 períodos
            5.0,    # ARG6: máximo valor do parâmetro
            10.0    # ARG7: máximo valor do parâmetro
        ]
    }
    
    # Criar problema de otimização
    problem = QBMDOptimizationProblem(config=config, bounds=bounds)
    
    # Configurar NSGA-II
    algorithm = NSGA2(
        pop_size=config["optimization"]["population_size"],
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=config["optimization"]["crossover_prob"], eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    # Critério de parada
    termination = get_termination("n_gen", config["optimization"]["num_generations"])
    
    # Executar otimização
    print("="*70)
    print("OTIMIZAÇÃO NSGA-II - QUANTUM BRAGG MIRROR DETECTOR")
    print("="*70)
    print(f"População: {config['optimization']['population_size']}")
    print(f"Gerações: {config['optimization']['num_generations']}")
    print(f"Total de simulações: {config['optimization']['population_size'] * config['optimization']['num_generations']}")
    print("="*70)
    
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=True,
        save_history=True
    )
    
    # Processar resultados
    print("\n" + "="*70)
    print("OTIMIZAÇÃO CONCLUÍDA")
    print("="*70)
    print(f"Soluções na Frente de Pareto: {len(res.X)}")
    
    # Converter objetivos de volta (remover negativo)
    pareto_objectives = -res.F
    
    # Salvar resultados
    save_results(res, pareto_objectives, config)
    
    # Mostrar melhores soluções
    display_best_solutions(res.X, pareto_objectives)
    
    # Visualizar Frente de Pareto
    visualize_pareto_front(pareto_objectives)
    
    return res


def save_results(res, pareto_objectives, config):
    """Salva os resultados da otimização"""
    
    output_dir = Path("./optimization_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar parâmetros (variáveis de decisão)
    params_file = output_dir / f"pareto_parameters_{timestamp}.csv"
    np.savetxt(
        params_file,
        res.X,
        delimiter=",",
        header="ARG1,ARG2,ARG3,ARG4,ARG5,ARG6,ARG7",
        comments=""
    )
    
    # Salvar objetivos
    objectives_file = output_dir / f"pareto_objectives_{timestamp}.csv"
    np.savetxt(
        objectives_file,
        pareto_objectives,
        delimiter=",",
        header="max_energy_eV,max_photocurrent,quality_factor,prominence",
        comments=""
    )
    
    # Salvar resultados combinados
    combined = np.hstack([res.X, pareto_objectives])
    combined_file = output_dir / f"pareto_full_{timestamp}.csv"
    np.savetxt(
        combined_file,
        combined,
        delimiter=",",
        header="ARG1,ARG2,ARG3,ARG4,ARG5,ARG6,ARG7,max_energy_eV,max_photocurrent,quality_factor,prominence",
        comments=""
    )
    
    print(f"\n Resultados salvos em: {output_dir}")
    print(f"   - Parâmetros: {params_file.name}")
    print(f"   - Objetivos: {objectives_file.name}")
    print(f"   - Completo: {combined_file.name}")


def display_best_solutions(parameters, objectives):
    """Mostra as melhores soluções por cada objetivo"""
    
    print("\n" + "="*70)
    print("MELHORES SOLUÇÕES POR OBJETIVO")
    print("="*70)
    
    objective_names = [
        "Energia do Pico (eV)",
        "Intensidade da Fotocorrente",
        "Fator de Qualidade",
        "Proeminência"
    ]
    
    for i, name in enumerate(objective_names):
        best_idx = np.argmax(objectives[:, i])
        
        print(f"\n🏆 Melhor {name}:")
        print(f"   Valor: {objectives[best_idx, i]:.6f}")
        print(f"   Parâmetros: {parameters[best_idx]}")
        print(f"   Outros objetivos:")
        for j, obj_name in enumerate(objective_names):
            if j != i:
                print(f"     - {obj_name}: {objectives[best_idx, j]:.6f}")


def visualize_pareto_front(objectives):
    """Visualiza projeções 2D da Frente de Pareto"""
    import matplotlib.pyplot as plt
    
    labels = [
        "Energia do Pico (eV)",
        "Intensidade",
        "Fator de Qualidade",
        "Proeminência"
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Todas as combinações de pares
    pairs = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3)
    ]
    
    for idx, (i, j) in enumerate(pairs):
        axes[idx].scatter(
            objectives[:, i], 
            objectives[:, j], 
            c='red', 
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        axes[idx].set_xlabel(labels[i], fontsize=11, fontweight='bold')
        axes[idx].set_ylabel(labels[j], fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_title(f"{labels[i]} vs {labels[j]}", fontsize=10)
    
    plt.tight_layout()
    
    output_file = 'pareto_front_projections.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualização da Frente de Pareto salva: {output_file}")
    
    plt.close()


if __name__ == "__main__":
    results = run_optimization()
