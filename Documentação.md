```markdown
# 1. Sumário Executivo

Desafio: como usar melhor o tempo da nossa equipe de negócios?  
Resposta: uma fila de priorização inteligente. 

Ao longo do projeto, não só construímos um modelo de Machine Learning, mas também respondemos à seguinte questão: queremos crescer a qualquer custo ou operar com máxima eficiência?

Testamos diferentes caminhos e chegamos a uma recomendação clara: adotar um **modelo Ensemble**, otimizado para eficiência. Ele garante que mais da metade dos clientes priorizados sejam leads de alta qualidade (precisão de 56%), aproveitando ao máximo o recurso mais caro da operação: o tempo da nossa equipe.

---

# 2. O Dilema Central: Crescimento vs. Eficiência

Esse projeto girou em torno de um trade-off clássico. Para entender a escolha, olhamos para duas métricas-chave:

- **Recall (Revocação):**  
  Mede quantos dos bons clientes o modelo consegue encontrar.  
  - Estratégia: *“Não perder oportunidades”*. Foco em crescimento.  
  - Melhor quando temos alta capacidade operacional e custo alto em perder clientes.

- **Precision (Precisão):**  
  Mede quantos dos clientes priorizados pelo modelo realmente são bons.  
  - Estratégia: *“Não desperdiçar tempo”*. Foco em eficiência.  
  - Melhor quando a equipe é enxuta e cada hora de consultor conta.

Nossa análise produziu dois modelos finalistas, cada um campeão em uma dessas frentes.

---

# 3. Os Dois Caminhos: Uma Análise Estratégica

### Modelo LightGBM — Crescimento
- **Recall:** 95%  
- **Precision:** 33%  
- **Impacto:** Ideal para crescimento acelerado. Garante quase todas as oportunidades, mas com baixa eficiência: de cada 3 ligações, 2 seriam para clientes de baixo potencial.

### Modelo Ensemble — Eficiência (Ensemble)
- **Recall:** 41%  
- **Precision:** 56%  
- **Impacto:** Ideal para operações otimizadas. Mais da metade das ligações são para bons clientes, aumentando muito a taxa de acerto e a produtividade da equipe.

---

# 4. Recomendação Estratégica: Escolhendo a Eficiência

Para o momento atual da Creditas, faz mais sentido apostar na eficiência.  
Nossa recomendação é implementar o **Modelo Ensemble**.

---

# 5. Interpretabilidade: Como o Modelo Vencedor “Pensa”?

O Ensemble não é uma caixa-preta. Ele aprendeu a dar peso aos mesmos sinais que um analista de crédito experiente avaliaria:

As variáveis mais relevantes foram:

- **collateral_debt (Dívidas com garantia):** O principal fator considerado. Representa o nível de endividamento vinculado aos veículos dados em garantia.  
- **total_debts (Dívidas Totais):** Captura a exposição global do cliente, ajudando a entender seu comprometimento financeiro.  
- **form_completed (Formulário completo):** Indica engajamento e seriedade do cliente no processo.
- **loan_to_value_ratio (Relação Empréstimo/Valor da Garantia):** Mede o quanto o valor solicitado compromete a garantia oferecida.
- **loan_to_income_ratio (Relação Empréstimo/Renda):** Mostra a alavancagem financeira do cliente.
- **monthly_payment (Parcela Mensal):** Ajuda a identificar se o compromisso mensal é viável dentro da renda declarada.  
- **monthly_income (Renda Mensal):** Ponto de partida para avaliar a capacidade de pagamento.  
- **collateral_value (Valor da Garantia):** Oferece segurança à operação e reduz risco em caso de inadimplência.  
- **informed_purpose (Finalidade do Empréstimo):** Traz contexto sobre a motivação da solicitação, diferenciando, por exemplo, necessidades urgentes de consumo de investimentos estruturados.

Esses fatores em conjunto permitem que o modelo priorize clientes com maior potencial de conversão e menor risco, trazendo eficiência e escala à análise — sem perder a lógica de negócio por trás da decisão.


---

# 6. O Valor do Machine Learning: Comparação com Benchmarks



| Método de Priorização                      | AUC   |
|--------------------------------------------|-------|
| Modelo Ensemble (Machine Learning)         | 0.812 |
| Modelo Simples de ML (Regressão Logística) | 0.755 |
| Benchmark 1 (Maior Renda)                  | 0.640 |
| Benchmark 2 (Menor Empréstimo)             | 0.580 |
| Benchmark 3 (Aleatório)                    | 0.500 |

**Conclusão:**  
O modelo de Machine Learning não só melhora o processo, como está em outro patamar. Ele enxerga relações complexas que regras simples nunca captariam.

---

# 7. Garantindo o Sucesso a Longo Prazo: Monitoramento

Um modelo é um ativo vivo. Para manter seu valor:

- **Performance do Modelo:**  
  - Monitorar semanalmente AUC e Precision.  
  - Analisar **drift** (mudança no perfil dos clientes).  

- **Impacto no Negócio:**  
  - Acompanhar taxa de conversão dos leads priorizados.  
  - Essa é a métrica final de sucesso e **ROI**.  

---

# 8. Melhorias Futuras

- **Novas Fontes de Dados:** Birôs de crédito para visão 360º.  
- **Interpretabilidade Individual (SHAP):** Explicações por cliente.  
- **Segmentação:** Modelos distintos para perfis diferentes (ex: CLT vs. autônomos).  

---

# 9. Conclusão

O **Modelo Ensemble** é a escolha certa para revolucionar a fila de priorização, trazendo mais eficiência e produtividade.  