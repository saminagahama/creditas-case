# Projeto de Priorização de Leads - Creditas Case

Esta pasta contém uma pipeline de Machine Learning para prever a probabilidade de um cliente ser enviado para análise de crédito, otimizando a fila de atendimento da equipe de negócios.

O projeto inclui notebooks para análise exploratória, uma pipeline de treinamento de modelos modularizada em scripts Python e scripts para realizar predições.

---

## Sumário
- [Visão Geral](#visão-geral)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Como Executar](#como-executar)
  - [Configuração do Ambiente](#1-configuração-do-ambiente)
  - [Treinamento do Modelo](#2-treinamento-do-modelo)
  - [Realizando Predições](#3-realizando-predições)
- [Opções Avançadas](#opções-avançadas)
  - [Alternando entre Modelos (Ensemble vs. LightGBM)](#alternando-entre-modelos-ensemble-vs-lightgbm)
  - [Ajustando o Threshold de Decisão](#ajustando-o-threshold-de-decisão)
  - [Otimização de Hiperparâmetros](#otimização-de-hiperparâmetros)
- [Entendendo as Métricas de Negócio](#entendendo-as-métricas-de-negócio)

---

## Visão Geral

O pipeline automatiza o processo de:

1. **Carregamento e Limpeza**: Lê dados brutos, trata valores faltantes e remove duplicatas.  
2. **Engenharia de Features**: Cria variáveis de negócio como `loan_to_income_ratio` e `total_debts`.  
3. **Treinamento**: Treina um dos dois modelos disponíveis (LightGBM ou Ensemble) e o salva como um arquivo `.pkl`.  
4. **Predição**: Carrega o modelo treinado para calcular a probabilidade de um novo cliente ser enviado para análise.  

---

## Estrutura do Projeto

```bash
creditas-case/
├── data/          # Dados brutos e processados
├── models/        # Modelos treinados (.pkl)
├── notebooks/     # Análise exploratória e prototipagem
├── src/           # Código fonte da pipeline
└── requirements.txt # Dependências do projeto

# Como Executar

## 1. Configuração do Ambiente

**Pré-requisitos:** Python 3.10+

Primeiro, crie um ambiente virtual e instale as dependências:

Na raiz do projeto:

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# Para Windows:
# venv\Scripts\activate
```

pip install --upgrade pip
pip install -r requirements.txt

## 2. Treinamento do Modelo

O script `train.py` permite treinar os modelos disponíveis. A forma mais simples é passando o nome do modelo como argumento:

- Treina e salva o modelo LightGBM em `models/modelo_lightgbm.pkl`:

```bash
python -m src.train lightgbm
```
Treina e salva o modelo Ensemble em models/modelo_ensemble.pkl:
```bash
python -m src.train ensemble
```
Ao final, o script exibirá as métricas de performance (AUC, Precision, Recall) e salvará o modelo treinado na pasta **models/**.

## 3. Realizando Predições

O script `predict.py` carrega um modelo treinado e o utiliza para pontuar um novo cliente.

Antes de rodar, certifique-se de que o modelo desejado está configurado em `src/config.py`:

```python
# src/config.py
MODEL_PATH = "models/modelo_ensemble.pkl"  # ou "models/modelo_lightgbm.pkl"
```

Execute a predição com o comando:
```bash
python -m src.predict
```

O script usará o cliente de exemplo definido no final de src/predict.py e imprimirá sua probabilidade.

Para uso em outras aplicações:
```bash
from src.predict import make_prediction
# Dicionário com dados do novo cliente...

novo_cliente = {'id': 99999, 'age': 35, 'pre_approved': True, ...}

resultado = make_prediction(novo_cliente)
print(f"Probabilidade: {resultado['probability']:.2%}")
```

## Alternando entre Modelos (Ensemble vs. LightGBM)

Para definir qual modelo será usado para predição ou qual será o padrão no treinamento, edite o arquivo `src/config.py` e ajuste a variável `MODEL_PATH`.

## Ajustando o Threshold de Decisão

Por padrão, uma probabilidade > 0.5 resulta em uma classificação positiva. Você pode ajustar este limiar (threshold) para otimizar para **Eficiência (Precision)** ou **Crescimento (Recall)**.

- Para testar um threshold na predição via linha de comando (ex.: usar um threshold de 0.3 para ser mais sensível, aumentando o Recall):

```bash
python -m src.predict 0.3
```

## Otimização de Hiperparâmetros

Os modelos já vêm com parâmetros pré-otimizados.

## Métricas e Negócio

* **Precision (Precisão):** De todos os clientes que o modelo disse que eram bons, quantos realmente eram? Esta é a métrica de "não desperdiçar tempo". Um threshold mais alto aumenta a precisão.

* **Recall (Revocação):** De todos os clientes que realmente eram bons, quantos o seu modelo conseguiu encontrar? Esta é a métrica de "não perder oportunidades". Um threshold mais baixo aumenta o recall.