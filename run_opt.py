#!/usr/bin/env python3
"""
Script principal para executar a otimização NSGA-II.
"""

import sys
from optimize import run_optimization

if __name__ == "__main__":
    try:
        print("Iniciando otimização multi-objetivo...")
        results = run_optimization()
        print("\n Otimização concluída com sucesso!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n  Otimização interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n Erro durante a otimização: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
