EXTRAÇÃO DE MÉTRICAS DE DESEMPENHO PARA DETECTORES QBMD
(Quantum Bragg Mirror Detectors)
Arquivo de entrada: Photocurrent_SL.txt


1. EFICIÊNCIA QUÂNTICA DE PICO (Peak QE) – Intensidade da fotocorrente

   Método: Extrair a fotocorrente máxima do arquivo Photocurrent_SL.txt

   Interpretação: Quanto maior o pico de fotocorrente, maior a
   eficiência quântica máxima do detector. Este valor representa a
   melhor conversão fóton-para-elétron alcançada pelo dispositivo. Essa curva nos fornece a informação de qual comprimento de onda gera a maior fotocorrente. Informações importantes para a gente: qual o comprimento de onda e qual a intensidade da corrente. Precisamos não apenas da intensidade máxima “np.max” mas também de qual o comprimento de onda em que esse máximo ocorre “np.where”. 

   Código: peak_QE = np.max(photocurrent)  


2. CORRENTE DE ESCURO (Dark Current) Corrente apenas do pico principal vs corrente total. Em outras palavras, percentual da corrente do pico principal vs toda fotocorrente.


   e se tiverem dois ou mais picos fora dessa faixa? como fica essa análise?

SUGESTÃO: Calcular a integral do pico principal da fotocorrente e dividir pela integral do módulo da fotocorrente total. Isso fornece uma ideia do percentual do pico principal frente à corrente total. 

   Código: proeminencia = np.mean(np.abs(photocurrent[energy < 0.1*peak_energy]))

   Interpretação: Menor corrente de escuro = melhor detector
   (menor ruído, maior razão sinal-ruído)

ATENÇÃO: isso não é corrente de escuro! 


3. EFICIÊNCIA QUÂNTICA INTEGRADA (Integrated QE) 

   Método: Calcular a área sob a curva de fotocorrente na banda de
   detecção, usando integração adaptativa baseada em threshold.

   Implementação:
   - Definir threshold = peak_value × 0.05 (5% do pico)
   - Selecionar apenas pontos onde photocurrent > threshold
   - Integrar a fotocorrente nesta região usando np.trapezoid()

   Código:
   threshold = peak_value * 0.05
   above_threshold = photocurrent > threshold
   QE_integrated = np.trapezoid(photocurrent[above_threshold],
                             energy[above_threshold])

   Interpretação: Representa a capacidade total de detecção através
   da faixa espectral. Maior área = melhor resposta integrada. A área da curva de interesse. É o oposto do que desejamos em um detector seletivo. Numa otimização desse parâmetro, queremos que ele seja o menor possível (em um detector seletivo). Talvez o melhor seria calcular o fator de qualidade do pico (valor do pico dividido pela largura a meia altura) como já está sendo feito no item 4.



4. SELETIVIDADE ESPECTRAL (Spectral Selectivity) – Na nossa área, chamamos isso de fator de qualidade. 

   Método: Calcular o FWHM (Full Width at Half Maximum) e dividir
   a energia de pico por este valor.

   Implementação:
   - Encontrar half_max = peak_value × 0.5
   - Identificar pontos onde photocurrent ≥ half_max
   - FWHM = E_direita - E_esquerda (largura em energia)
   - Fator de qualidade = ENERGIA DO PICO / FWHM

   Código:
   half_max = peak_value * 0.5
   above_half = photocurrent >= half_max
   indices = np.where(above_half)[0]
   fwhm = energy[indices[-1]] - energy[indices[0]]
   Fator de qualidade= Energia do pico / fwhm

   Unidades: (a.u.) / eV

   Interpretação:
   - Alta seletividade  <0.15): pico estreito, excelente discriminação
   - Baixa seletividade ( >0.3): pico largo, detecção banda-larga


RESUMO:
Total de otimizações:
- Qual a energia do pico principal? (idealmente apenas um pico principal)
- Qual a intensidade da fotocorrente do pico principal?
- Qual o fator de qualidade do pico principal?
- Qual o percentual da corrente do pico principal frente à corrente total?