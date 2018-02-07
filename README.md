
Projeto 5 - Identificar fraude no Email da Enron 
====================
##### Por Nikolas Thorun

Em 2001 a Enron Corporation foi à falência. Era uma das líderes mundiais no fornecimento de energia e serviços. Durante anos, diretores da empresa maquiavam os balancetes, enxugavam os prejuízos e inflavam os lucros. Denúncias e rumores promoveram o que seria o maior escândalo financeiro dos Estados Unidos.
O objetivo deste projeto é analisar características dos funcionários da empresa a fim de conseguir predizer quais deles são Pessoas de Interesse (POI), ou seja, funcionários que participaram da fraude. O uso do aprendizado de máquina é vantajoso neste caso, pois torna o processo de processamento de dados muito mais rápido e eficiente do que um cérebro humano.

#### Visão geral dos dados

O conjunto de dados consiste em 146 registros com 20 características (_features_) e 1 rótulo (_label_) (POI). Algumas características são financeiras, outras dizem respeito ao uso de e-mails. 18 registros são rotulados como POI's, pessoas que estavam comprovadamente envolvidos na fraude.
Durante as investigações iniciais, 2 dos 146 registros não representavam pessoas, por isso foram removidos.


* `'TOTAL'`
* `'THE TRAVEL AGENCY IN THE PARK'`





#### Seleção de Características
Como padrão, o SelectKBest calcula os valores-F da ANOVA (Análise de Variância), ou seja, podemos usá-lo para selecionar as características de maior variância. 
Através desse algoritmo, foram selecionadas as 12 características iniciais de maior variância. As características e os valores-F são mostrados na tabela abaixo.
        
|Features | Weight | 
|:-------|:------:|
|  exercised_stock_options | 24.81  |
|  total_stock_value | 24.18  |
|  bonus | 20.79  | 
|  salary | 18.28  |
|  deferred_income | 11.45 |
|  long_term_incentive | 9.92  |
|  restricted_stock | 9.21  |
|  total_payments | 8.77  |
|  shared_receipt_with_poi | 8.58  |
|  loan_advances | 7.18  |
|  expenses | 6.09  |
|  from_poi_to_this_person | 5.24  |

Através de tentativa e erro, o número de características foi escolhido por apresentar melhores métricas de avaliação.

Nenhum processo de escalonamenteo foi utilizado, tendo em vista que os algoritmos testados não utilizam a distância euclidiana para calcular a distância entre dois pontos. Em casos como o do algoritmo KNN, por exemplo, se o conjunto de dados não for escalonado, as variáveis que apresentarem maiores valores terão mais influência no classificador. O escalonamento visa criar variáveis não dimensionais, de maneira que a grandeza dessas não enviezem o resultado. 

Duas novas variáveis foram criadas e utilizadas juntamente com as 12 características originais selecionadas.
Elas são: `'messages_from_poi_ratio'` e `'messages_to_poi_ratio'`. A idéia é saber qual é a proporção em que um determinado funcionário recebe e envia e-mails para algum POI. 


#### Classificador Selecionado
O Algoritmo selecionado ao final dos testes foi o AdaBoost utilizando o Decision Trees como estimador básico. Além dele, foram testados o Naive Bayes, SVM, Decision Tree, Random Forest e Gradient Boosting. AdaBoost e Decision Tree foram os que obtiveram maiores valores do F1 score. 
Os resultados são mostrados abaixo com a segunda casa decimal arredondada.


| | AdaBoost | DecisionTree   |
|:-------|:------:|:------:|
|  Acurácia | 0.85  | 0.88 |
|  Precisão | 0.45  | 0.57 |
|  Revocação | 0.62  | 0.44 |
|  F1 score | 0.52  | 0.50 |

Os demais algoritmos não passaram de 0.40 no F1 score.

#### Ajustes

Os ajustes finos no algoritmo desempenham a função de otimizar a performance do classificador. Os classificadores, sem parâmetros definidos, tendem a ser generalistas e, para ajustá-los a um problema específico, se fazem necessários ajustes. Caso esse processo seja mal feito, o classificador está sujeito a uma performance ruim no conjunto de treino (_underfitting_) ou a ficar específico demais (_overfitting_).
No início do projeto o GridSearchCV foi utilizado para obter os parâmetros ótimos do classificador. Porém, ele retorna os parâmetros que obtiveram maior acurácia durante a validação cruzada e, neste caso, seria interessante saber se existe uma combinação de parâmetros que possui a acurácia um pouco menor e uma taxa de revocação bastante maior, por exemplo.
Por isso, a função `product` do módulo `itertools` foi utilizada de maneira a gerar todas as combinações possíveis entre os parâmetros. Para cada combinação, a função `test_classifier` do `tester.py` é chamada e imprime os resultados no console.
Os parâmetros ajustados foram `n_estimators` (do 1 ao 5), `learning_rate` (do 1 ao 3), `max_depth` (do 1 ao 3), `min_samples_split` (do 2 ao 5) e `min_samples_leaf` (do 1 ao 10, de 2 em 2). Os dois primeiros são parâmetros do AdaBoost e o restante pertence ao Decision Tree, que é seu estimador básico.


#### Validação

Validação é o processo de testar o algoritmo de aprendizado de máquina em dados que não foram utilizados durante a etapa de treinamento. Pode-se dividir um conjunto de dados em dois, usar uma parte para treinar o algoritmo e outra parte para validar o quão bom foram os resultados previstos pelo algoritmo comparando-os com os valores reais deste sub-conjunto. Uma maneira clássicade errar durante este processo é utilizar o conjunto inteiro de dados para o treinamento e depois testar o algoritmo em parte desse conjunto. O classificador provavelmente será sobreajustado e proporcionará resultados ruins em dados novos.
A validação utilizada nesse projeto foi a da função `test_classifier` do `tester.py`. Nessa função é utilizado o StratifiedShuffleSplit que recebe o conjunto de dados e o divide em conjuntos de treinamento e teste por mil vezes, na proporção de 90% para 10%. Os resultados obtidos ao final são as médias dos resultados em cada divisão.

#### Métricas de avaliação
Para o classificador final foram utilizadas quatro métricas de avaliação:

| | AdaBoost | 
|:-------|:------:|
|  Acurácia | 0.85  | 
|  Precisão | 0.45  | 
|  Revocação | 0.62  | 
|  F1 score | 0.52  | 

* Acurácia

Pode-se dizer que aproximadamente 85% das observações foram classificadas corretamente entre POI e não-POI. Porém, sabendo que apenas 18 dos 144 (12,5%) empregados são POI's, se o modelo classificasse todos funcionários como não-POI o classificador teria 87,5% de acurácia. Ou seja, acurácia é uma métrica importante mas não seria inteligente usá-la apenas.
* Precisão

Pode-se dizer que aproximadamente 45% das observações classificadas como POI's eram realmente POI's. Ou seja, 55% das observações classificadas como POI's são falsos positivos. Quanto maior a precisão, menos funcionários serão falsamente acusados de serem POI's.
* Revocação (_Recall_)

Pode-se dizer que aproximadamente 62% das observações que realmente são POI's foram corretamente identificados. Ou seja, 38% dos POI's são falsos negativos. Quanto maior for a revocação, menor é a probabilidade de algum POI escapar de sua identificação.
* F1 score

O F1 score é a combinação da precisão e da revocação de maneira que com apenas um número, sabemos o quão bem o classificador trabalha. Quanto maior for o F1 score, melhor.
Por isso o AdaBoost foi escolhido, pois apesar de apresentar acurácia menor que o Decision Tree, o F1 score foi ligeiramente superior. Além disso, a revocação do AdaBoost foi bastante superior, diminuindo a possibilidade de um POI se safar.
Nesse projeto, buscar maiores valores de revocação em detrimento de menores valores de precisão é compreensível, uma vez que identificar erroneamente um POI não é tão ruim quanto deixar de identificar um POI.

