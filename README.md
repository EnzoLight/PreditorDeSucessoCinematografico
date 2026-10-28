Preditor de Sucesso Cinematográfico (PSC)
Instituição: FATEC Mauá

Curso: Desenvolvimento de Software Multiplataforma

Disciplina: MAQ-024 — Aprendizagem de Máquina

Professor: Bruno Zolotareff dos Santos
Integrantes
Marco Aurelio Dias Torres Machado
Mauro Luiz dos Santos Junior
Vitor Aletto D'Andrea
Gabriel Menezes da Silva
Enzo Luz Granato Barbosa
1. Objetivo e problema
O projeto tem como objetivo desenvolver um modelo de Machine Learning para classificar filmes de acordo com seu potencial de sucesso financeiro. A proposta aborda a incerteza envolvida no desempenho comercial de produções cinematográficas e investiga como dados históricos podem apoiar essa análise.
Para o MVP, um filme é considerado sucesso quando sua bilheteria registrada na base do TMDB é igual ou superior a duas vezes seu orçamento de produção:
Sucesso = bilheteria ≥ 2 × orçamento
O modelo recebe características disponíveis na base de dados, como ano de lançamento, duração, orçamento, gêneros, participação em franquia e informações relacionadas à frequência de aparição do diretor e do roteirista na amostra.
2. Solução e funcionamento do MVP
O MVP é desenvolvido em Python, no Google Colab. O código carrega a base pública de filmes da UCSB, seleciona os registros classificados como filmes e consulta a API do TMDB para obter orçamento, bilheteria, informação de franquia, diretor e roteirista. Os dados enriquecidos são salvos em um arquivo CSV de cache para evitar repetir as consultas em execuções posteriores no mesmo ambiente.
Na preparação dos dados, o código remove registros com orçamento ou bilheteria inferiores ou iguais a mil. Em seguida, cria a variável de sucesso conforme o critério definido. Os gêneros são convertidos em variáveis indicadoras, e as informações de diretor e roteirista são representadas pela frequência com que aparecem na amostra.
O modelo utilizado nesta versão é o XGBoost Classifier, configurado com 200 estimadores, taxa de aprendizado de 0,05 e profundidade máxima de 5. O conjunto de dados é ordenado pelo ano de lançamento e dividido cronologicamente: os primeiros 80% são usados no treinamento e os 20% finais, no teste.
A avaliação é feita por meio de acurácia, precisão, recall, F1-score e matriz de confusão. Na implementação atual, o modelo retorna uma classificação — sucesso ou não sucesso — e não uma probabilidade individual para cada filme.
3. Base de dados e ferramentas
Base inicial: Streaming Dataset — UCSB
Enriquecimento: API do TMDB
Ambiente: Google Colab
Linguagem: Python
Bibliotecas principais: XGBoost, pandas, NumPy, requests, tqdm e scikit-learn
Notebook do MVP: Preditor de bilheteria — Google Colab
4. Limitações e próximos ajustes
O código atual deve ser entendido como um MVP experimental. A busca no TMDB usa o título e o ano para localizar o filme e seleciona o primeiro resultado encontrado, o que pode associar alguns registros ao filme errado. Além disso, as frequências de diretores e roteiristas são calculadas antes da divisão entre treino e teste, usando toda a amostra. Isso pode introduzir informação dos dados de teste no treinamento e comprometer a avaliação cronológica. Para reduzir esse risco, essas frequências devem ser calculadas usando somente os dados de treino e aplicadas ao conjunto de teste sem consultar seus registros futuros.
A descrição do projeto também deve refletir que o Random Forest “from scratch”, a comparação com um baseline separado, a extração de importância das variáveis e a previsão probabilística não fazem parte do código atual. Esses itens podem ser apresentados como possíveis evoluções, caso sejam implementados posteriormente.
5. Referências
University of California, Santa Barbara. Streaming Dataset — OpenRefine. [Acessar](https://rds.library.ucsb.edu/openrefine/dataset.html)
The Movie Database (TMDB). Documentação da API. [Acessar](https://developer.themoviedb.org/docs/search-and-query-for-details)
XGBoost. Documentação oficial. [Acessar](https://xgboost.readthedocs.io/en/stable/)
