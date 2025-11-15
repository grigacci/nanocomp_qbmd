"""
Otimização multi-objetivo de detectores QBMD usando NSGA-II.
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    - RW: Número de poços à esquerda (int)
    - RQWt: Espessura dos poços quânticos à esquerda (nm) (float)
    - RQBt: Espessura das barreiras à esquerda (float)
    - MQWt: Espessura do poço quântico principal (float)
    - LW: Número de poços à direita (int)
    - LQWt: Espessura dos poços quânticos à direita (nm) (float)
    - LQBt: Espessura das barreiras à direita (nm)
    
    Objetivos (maximizar):
    1. Energia do pico principal
    2. Intensidade da fotocorrente
    3. Fator de qualidade
    4. Proeminência do pico
    
    Restrições:
    1. MQWt >= min(RQWt, LQWt) + 2*monocamada
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
        self.base_output_dir = Path(config["paths"]["output_dir"])
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Obter espessura da monocamada da configuração (com valor padrão)
        self.monolayer_thickness = config["optimization"].get("monolayer_thickness", 0.3)
        
        # Definir limites das variáveis
        xl = np.array(bounds['lower'])
        xu = np.array(bounds['upper'])
        
        # 7 variáveis, 4 objetivos, 1 restrição
        super().__init__(
            n_var=7,
            n_obj=4,
            n_ieq_constr=1,  
            xl=xl,
            xu=xu
        )

    def _run_simulation_with_id(self, sim_id, individual):
        # Build a unique per-sim output path using sim_id (not a shared counter)
        RW = int(individual[0])
        RQWt = f"{individual[1]:.1f}d0"
        RQBt = f"{individual[2]:.1f}d0"
        MQWt = f"{individual[3]:.1f}d0"
        LW = int(individual[4])
        LQWt = f"{individual[5]:.1f}d0"
        LQBt = f"{individual[6]:.1f}d0"
        simulation_path = f"{RW:02d}x{RQWt}_{RQBt}_{MQWt}_{LW:02d}x{LQWt}_{LQBt}"
        output_dir = self.base_output_dir / f"sim_{sim_id:05d}" / simulation_path
        output_dir.mkdir(parents=True, exist_ok=True)

        # Optional: set OpenMP/BLAS threads to avoid oversubscription
        env = os.environ.copy()
        omp = str(self.config["simulator"].get("omp_threads", 1))
        env.setdefault("OMP_NUM_THREADS", omp)
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")

        args = [
            str(self.config["simulator"]["exec_path"]),
            str(RW), RQWt, RQBt, MQWt,
            str(LW), LQWt, LQBt,
            str(output_dir)
        ]
        try:
            result = subprocess.run(
                args, capture_output=True, text=True,
                timeout=self.config["simulator"]["timeout"], env=env
            )
            if result.returncode != 0:
                return self._default_result(), individual
            photocurrent_file = output_dir / self.config["paths"]["photocurrent"]
            if not photocurrent_file.exists():
                return self._default_result(), individual
            simr = getSimulationResult(str(photocurrent_file),
                                    self.config["analysis"]["peak_threshold"])
            return simr, individual
        except Exception:
            return self._default_result(), individual

    
     
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Avalia a população de soluções.
        
        Parâmetros
        ----------
        X : array (pop_size, 7)
            Matriz com os indivíduos da população
        out : dict
            Dicionário para armazenar os objetivos e restrições
        """
        n = len(X)
        start_id = self.simulation_counter + 1
        sim_ids = [start_id + i for i in range(n)]
        F, G = [None]*n, [None]*n
    
        # Pool size from config or CPU
        max_workers = int(self.config["optimization"].get("n_workers", os.cpu_count()))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self._run_simulation_with_id, sim_ids[i], X[i]): i for i in range(n)}
            for fut in as_completed(futures):
                i = futures[fut]
                res, individual = fut.result()
                obj = [-res.max_energy, -res.max_photocurrent, -res.quality_factor, -res.prominence]
                RW, RQWt, RQBt, MQWt, LW, LQWt, LQBt = individual
                constraint_value = min(RQWt, LQWt) - MQWt + 2 * self.monolayer_thickness
                F[i] = obj
                G[i] = [constraint_value]
    
        self.simulation_counter += n
        out["F"] = np.array(F)
        out["G"] = np.array(G) 
    
    def _run_simulation(self, parameters):
        """
        Executa o simulador QBMD com os parâmetros fornecidos.
        
        Parâmetros
        ----------
        parameters : array [RW, RQWt, RQBt, MQWt, LW, LQWt, LQBt]
        
        Retorna
        -------
        SimulationResult
        """
        
        # Converter parâmetros para formato correto
        RW = int(parameters[0])  # Inteiro
        RQWt = f"{parameters[1]:.1f}d0"  # Formato Fortran
        RQBt = f"{parameters[2]:.1f}d0"
        MQWt = f"{parameters[3]:.1f}d0"
        LW = int(parameters[4])  # Inteiro
        LQWt = f"{parameters[5]:.1f}d0"
        LQBt = f"{parameters[6]:.1f}d0"

        simulation_path = f"{RW:02d}x{RQWt}_{RQBt}_{MQWt}_{LW:02d}x{LQWt}_{LQBt}"

        # Diretório de saída específico para esta simulação
        output_dir = self.base_output_dir / f"sim_{self.simulation_counter:05d}" / simulation_path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Executar script run.sh
        script_path = self.config["simulator"]["exec_path"]
        
        args = [
                    str(script_path),
                    str(RW), RQWt, RQBt, MQWt,
                    str(LW), LQWt, LQBt,
                    str(output_dir)
                ]
        
        print(f"\n Iniciando simulação {self.simulation_counter} com parâmetros: {args}")

        try:
            result = subprocess.run(
                args,
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
    
    # Obter bounds do arquivo config.toml
    bounds_config = config["optimization"]["bounds"]
    
    # Construir dicionário de limites na ordem das variáveis:
    # [RW, RQWt, RQBt, MQWt, LW, LQWt, LQBt]
    bounds = {
        'lower': [
            bounds_config["rw_min"],      # RW: Número de poços à esquerda
            bounds_config["rqwt_min"],    # RQWt: Espessura dos poços quânticos à esquerda
            bounds_config["rqbt_min"],    # RQBt: Espessura das barreiras à esquerda
            bounds_config["mqwt_min"],    # MQWt: Espessura do poço quântico principal
            bounds_config["lw_min"],      # LW: Número de poços à direita
            bounds_config["lqwt_min"],    # LQWt: Espessura dos poços quânticos à direita
            bounds_config["lqbt_min"]     # LQBt: Espessura das barreiras à direita
        ],
        'upper': [
            bounds_config["rw_max"],      # RW: Número de poços à esquerda
            bounds_config["rqwt_max"],    # RQWt: Espessura dos poços quânticos à esquerda
            bounds_config["rqbt_max"],    # RQBt: Espessura das barreiras à esquerda
            bounds_config["mqwt_max"],    # MQWt: Espessura do poço quântico principal
            bounds_config["lw_max"],      # LW: Número de poços à direita
            bounds_config["lqwt_max"],    # LQWt: Espessura dos poços quânticos à direita
            bounds_config["lqbt_max"]     # LQBt: Espessura das barreiras à direita
        ]
    }
    
    # Restante do código permanece igual...
    problem = QBMDOptimizationProblem(config=config, bounds=bounds)
    
    algorithm = NSGA2(
        pop_size=config["optimization"]["population_size"],
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=config["optimization"]["crossover_prob"], eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    termination = get_termination("n_gen", config["optimization"]["num_generations"])
    
    print("="*70)
    print("OTIMIZAÇÃO NSGA-II - QUANTUM BRAGG MIRROR DETECTOR")
    print("="*70)
    print(f"População: {config['optimization']['population_size']}")
    print(f"Gerações: {config['optimization']['num_generations']}")
    print(f"Total de simulações: {config['optimization']['population_size'] * config['optimization']['num_generations']}")
    print(f"Espessura monocamada: {config['optimization'].get('monolayer_thickness', 0.3)} nm")
    print("="*70)
    
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=True,
        save_history=True
    )
    
    # Processar resultados...
    print("\n" + "="*70)
    print("OTIMIZAÇÃO CONCLUÍDA")
    print("="*70)
    print(f"Soluções na Frente de Pareto: {len(res.X)}")
    
    # Verificar violações de restrição nas soluções finais 
    violations = res.G.max(axis=0) if hasattr(res, 'G') and res.G is not None else [0]
    print(f"Máxima violação de restrição: {violations[0]:.6f} (deve ser ≤ 0)")
    
    pareto_objectives = -res.F
    
    save_results(res, pareto_objectives, config)
    
    return res


def save_results(res, pareto_objectives, config):
    """Salva os resultados da otimização"""
    
    config_paths = config["paths"]["result_dir"]

    output_dir = Path(config_paths)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar parâmetros (variáveis de decisão)
    params_file = output_dir / f"pareto_parameters_{timestamp}.csv"
    np.savetxt(
        params_file,
        res.X,
        delimiter=",",
        header="RW,RQWt,RQBt,MQWt,LW,LQWt,LQBt",
        comments="",
        fmt="%d,%.6f,%.6f,%.6f,%d,%.6f,%.6f"  
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
    
    # Salvar restrições
    constraints_file = output_dir / f"pareto_constraints_{timestamp}.csv"
    if hasattr(res, 'G') and res.G is not None:
        np.savetxt(
            constraints_file,
            res.G,
            delimiter=",",
            header="min_lateral_minus_MQWt_plus_2monolayer",
            comments=""
        )
    
    # Salvar resultados combinados
    if hasattr(res, 'G') and res.G is not None:
        combined = np.hstack([res.X, pareto_objectives, res.G])
        header_combined = "RW,RQWt,RQBt,MQWt,LW,LQWt,LQBt,max_energy_eV,max_photocurrent,quality_factor,prominence,constraint"
    else:
        combined = np.hstack([res.X, pareto_objectives])
        header_combined = "RW,RQWt,RQBt,MQWt,LW,LQWt,LQBt,max_energy_eV,max_photocurrent,quality_factor,prominence"
    
    combined_file = output_dir / f"pareto_full_{timestamp}.csv"
    np.savetxt(
        combined_file,
        combined,
        delimiter=",",
        header=header_combined,
        comments="",
        fmt="%d,%.6f,%.6f,%.6f,%d,%.6f,%.6f,%.6f,%.6e,%.6f,%.6f,%.6f"
    )
    
    print(f"\n Resultados salvos em: {output_dir}")
    print(f"   - Parâmetros: {params_file.name}")
    print(f"   - Objetivos: {objectives_file.name}")
    print(f"   - Restrições: {constraints_file.name}")  
    print(f"   - Completo: {combined_file.name}")





if __name__ == "__main__":
    results = run_optimization()