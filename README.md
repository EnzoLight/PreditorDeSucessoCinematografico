# Preditor de Sucesso Cinematográfico (PSC)

Projeto acadêmico da FATEC Mauá para explorar o uso de aprendizado de máquina na classificação do potencial de sucesso financeiro de filmes.

## Sobre o projeto

A indústria cinematográfica envolve incerteza financeira: uma produção pode exigir um investimento elevado sem recuperar seus custos nas bilheterias. Este projeto investiga como dados históricos podem ajudar a classificar filmes de acordo com seu desempenho comercial.

Para o MVP, um filme é considerado **sucesso** quando a bilheteria registrada no TMDB é igual ou superior a duas vezes seu orçamento de produção:

```text
Sucesso = bilheteria >= 2 * orçamento
```

O resultado produzido pela implementação atual é uma classificação entre sucesso e não sucesso. O modelo ainda não calcula uma probabilidade individual de sucesso.

## Instituição e equipe

- **Instituição:** FATEC Mauá
- **Curso:** Desenvolvimento de Software Multiplataforma
- **Disciplina:** MAQ-024 — Aprendizagem de Máquina
- **Professor:** Bruno Zolotareff dos Santos

### Integrantes

- Marco Aurelio Dias Torres Machado
- Mauro Luiz dos Santos Junior
- Vitor Aletto D'Andrea
- Gabriel Menezes da Silva
- Enzo Luz Granato Barbosa

## MVP

O notebook é desenvolvido em Python no Google Colab. O processo implementado consiste em:

1. Carregar a base pública de filmes da UCSB e selecionar os registros classificados como filmes.
2. Consultar a API do TMDB para obter orçamento, bilheteria, indicação de franquia, diretor e roteirista.
3. Salvar os dados enriquecidos em um arquivo CSV de cache para reaproveitá-los em execuções posteriores no mesmo ambiente.
4. Remover registros com orçamento ou bilheteria inferiores ou iguais a mil.
5. Criar o rótulo de sucesso conforme o critério de bilheteria igual ou superior a duas vezes o orçamento.
6. Preparar as variáveis: converter gêneros em indicadores e representar diretor e roteirista pela frequência com que aparecem na amostra.
7. Ordenar os dados pelo ano de lançamento e separar os primeiros 80% para treinamento e os 20% finais para teste.
8. Treinar e avaliar um classificador XGBoost.

### Variáveis utilizadas

O conjunto de entrada do modelo inclui:

- Ano de lançamento;
- Duração;
- Orçamento;
- Indicador de participação em franquia;
- Frequência do diretor na amostra;
- Frequência do roteirista na amostra;
- Gêneros do filme, convertidos em variáveis indicadoras.

A bilheteria é usada para definir o rótulo de sucesso e não é incluída como variável de entrada do modelo.

### Modelo e avaliação

O notebook utiliza `XGBClassifier` com os seguintes parâmetros:

- `n_estimators=200`;
- `learning_rate=0.05`;
- `max_depth=5`;
- `eval_metric="logloss"`;
- `random_state=42`.

O desempenho é avaliado por acurácia, precisão, recall, F1-score e matriz de confusão, usando métricas do scikit-learn.

## Base de dados e ferramentas

- **Base inicial:** [Streaming Dataset — UCSB](https://rds.library.ucsb.edu/openrefine/data/master-streaming-messy.csv )
- **Dados complementares:** [API do TMDB](https://developer.themoviedb.org/docs/search-and-query-for-details )
- **Ambiente:** [Google Colab](https://colab.research.google.com/drive/1WPqqDX0WjQaHrtDNe6I4coG7tSbwtc3r?usp=sharing)
- **Linguagem:** Python
- **Bibliotecas:** XGBoost, scikit-learn, pandas, NumPy, requests e tqdm

## Limitações conhecidas

- A correspondência de filmes no TMDB é feita com título e ano, e o código usa o primeiro resultado da busca. Portanto, alguns registros podem ser associados a um filme incorreto.
- As frequências de diretores e roteiristas são calculadas antes da divisão entre treino e teste, considerando toda a amostra. Isso pode introduzir vazamento de informação. Uma versão futura deve calcular essas frequências apenas no conjunto de treino e aplicar o mapeamento ao conjunto de teste.
- A divisão cronológica ordena os registros pelo ano de lançamento, mas pode separar filmes do mesmo ano entre treino e teste.
- O modelo atual não implementa Random Forest do zero, comparação com um baseline separado, importância das variáveis nem classificação probabilística.
- O critério de sucesso é uma simplificação baseada na relação entre bilheteria e orçamento. Ele não inclui outros custos, como marketing, distribuição ou participação de exibidores.
- Valores ausentes, incorretos ou indisponíveis na base da UCSB ou na API do TMDB podem afetar os resultados.


## Referências

1. University of California, Santa Barbara. [Streaming Dataset — OpenRefine](https://rds.library.ucsb.edu/openrefine/dataset.html ).
2. The Movie Database (TMDB). [Documentação da API](https://developer.themoviedb.org/docs/search-and-query-for-details ).
3. XGBoost. [Documentação oficial](https://xgboost.readthedocs.io/en/stable/ ).
4. Breiman, L. (2001). [Random Forests](https://doi.org/10.1023/A:1010933404324 ). *Machine Learning*, 45, 5–32. Referência de contexto sobre métodos de ensemble; o notebook atual não implementa Random Forest.

## Implementação alternativa: NumPy + chat com Ollama

Esta implementação nova foi adicionada como uma alternativa didática ao MVP original descrito acima. **O README histórico e os arquivos de dados existentes foram mantidos**: o código original baseado em XGBoost/scikit-learn não foi substituído por esta versão.

### Arquivos adicionados

- `preditor_sucesso.py`: preditor por regressão logística implementada com NumPy; usa pandas para carregar/preparar a base, sem scikit-learn no treinamento ou nas métricas.
- `PreditorSucessoCinematografico_Ollama.ipynb`: notebook para Colab que instala e inicia Ollama, baixa `lfm3:1.7b`, executa treinamento e conecta um chat local.
- `requirements.txt`: dependências Python da nova implementação.
- `tests/test_preditor_sucesso.py`: testes do treinamento, métricas, validação, extração e roteamento de conversa.

O critério de sucesso continua sendo `revenue >= 2 * budget`; é uma simplificação e não significa lucro. A nova versão faz divisão cronológica por anos, aprende transformações apenas no treino e reporta métricas no teste mais recente. As probabilidades não são calibradas e não garantem resultados futuros.

### Executar a alternativa no Colab

Abra `PreditorSucessoCinematografico_Ollama.ipynb` e rode as células em ordem. Na etapa de upload, envie `base_enriquecida_tmdb.csv` e `preditor_sucesso.py`. O notebook baixa o modelo local `lfm3:1.7b` (aproximadamente 1,1 GB); recomenda-se uma sessão Colab com GPU. Não é necessário compartilhar uma chave de API.

O chat usa a LLM para interpretar os dados e conversar, mas somente o código Python valida os campos e calcula a previsão. Conversas e dúvidas sobre resultados não devem alterar a estimativa já calculada.

Para verificar o código localmente:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```
