# PROJETO SUPER MARIO WORLD – YOSHI ISLAND 2

### Instruções de execução do projeto e arquivos necessários para a execução de um agente inteligente de treino cujo objetivo é completar o nível Yoshi Island 2 do jogo Super Mario World utilizando a biblioteca NEAT.

#### O modelo foi treinado somente nesta fase e pode não apresentar bom desempenho em demais níveis.

## Integrantes

- Camila Lorena Ferreira dos Santos (11201920711)
- Fabio Ferreira da Silva (11202231452)
- João Pedro Genga Carneiro (11201810740)
- Nicolas Marcelino da Silva (11202021107)
- Victor Inácio da Silva (11201810048)
- Wellington Pereira Trindade (11202020121)
- William Fernandes Dias (11202020043)

## Requisitos para execução

Python 3.8._ instalado e as seguintes bibliotecas **em um ambiente virtual**:

* neat
* numpy
* os
* argparse
* retro
* random
* pickle
* gzip
* time
* gym==0.21.0
* gym-retro

> Copie a ROM do jogo para uma pasta específica da biblioteca para que seja possível executá-la:
> 
> ```bash
> cp rom.sfc marioenv\Lib\site-packages\retro\data\stable\SuperMarioWorld-Snes
> ```

### 5. Executando o programa

* Para iniciar um  treinamento novo:

```bash
python mario_train.py novo new_train None
```

Onde "Treinamento_Novo" o nome de um diretório não existente onde os checkpoints geracionais e melhores genomas de cada geração serão armazenados, e None indica não existir um arquivo a partir do qual continuar o treino.

* Para continuar o treinamento de onde ele parou:

```bash
python mario_train.py continuar Train neat-checkpoint-generation-170.pkl
```

Onde "continuar" é o modo de treinamento, "Treinamento" é o nome de um diretório existente, de onde o arquivo de checkpoint deve estar, e "neat-checkpoint-185" é o arquivo de checkpoint.

* Para executar o melhor modelo alcançado pelo grupo:

```bash
python play.py best_model_distance/"neat-checkpoint-generation-- melhor genoma generation - 170.pkl"
```

## Funcionamento geral e bibliotecas

### 1. NEAT

Para realizar o treinamento do agente foi utilizada biblioteca NEAT (acrônimo do inglês "NeuroEvolution of Augmenting Topologies"), a qual permite a criação de uma rede neural artificial. Em resumo, o algoritmo baseia-se na criação de populações de genomas, onde cada um destes possui nós e conexões as quais permitem o agente se aprimorar, "aprendendo". O usuário ainda deve fornecer uma função indicando qual parâmetro deve ser utilizado como medida de qualidade do genoma processado (fitness, um valor do tipo numérico). Cada genoma tem uma chance de completar o nível e à ele é atruibuido o valor de aptidão a partir da função escolhida. O processo é repetido por um número específico de gerações, até que o número de gerações definido seja alcançado ou a função de fitness seja ultrapassada. Como resultado destas repetições, são modificados nós e conexões na rede neural através de mutações com o objetivo de chegar ao resultado esperado. A documentação oficial da biblioteca e mais detalhes sobre suas funcionalidades disponíveis podem ser encontradas [aqui](https://neat-python.readthedocs.io/en/latest/).

### 2. Funcionamento geral da implementação

O algoritmo utilizado foi de neuroevolução, com auxílio da biblioteca de python NEAT, com funções nativas da biblioteca que permitem a implementação de uma [rede neural recorrente](https://neat-python.readthedocs.io/en/latest/module_summaries.html#nn-recurrent) a partir do genoma em execução e as configurações definidas. O arquivo é composto de basicamente três funções e uma classe:

* #### função `calculate_genome_fitness`

Função que  calcula e define o valor de fitness com base na distância percorrida em determinada ação tomada pelo agente para cada genoma que recebe como parâmetro (presentes na variável genome). Nesta etapa são feitas também verificações que podem impor condições de reinicialização da execução, como se o agente chegou ao objetivo da fase, se morreu ou se ficou parado durante muito tempo em determinado estado.

* #### função `novoTreinamento`

Caso o usuário deseje iniciar um novo treino a partir do 0, essa função será chamada definindo uma nova população a partir do que está definido em config-feedforward e adiciona as estatísticas referentes a ela. Após todos cálculos com os genomas serem realizados, salva e cria uma nova rede neural com o modelo que obteve maior sucesso naquela execução. A função ainda recebe o nome de uma pasta a ser criada, para salvar os arquivos de checkpoint. Caso o nome recebido já exista, o comando de input entra em loop até receber o nome de uma pasta que não exista para prosseguir com a criação.

* #### função `continuarTreinamento`

Permite que o usuário escolha continuar o treino do agente a partir de um checkpoint salvo de treinos anteriores. Para isso, precisa do diretório onde os arquivos estão salvos e o último checkpoint disponível (ambos passados como parâmetros). As demais funcionalidades são análogas às presentes na novoTreinamento.

* #### classe `MeuCheckpointer`

Classe responsável por salvar o andamento do treino executado, disponível [aqui](https://github.com/CodeReclaimers/neat-python/blob/master/neat/checkpoint.py). Adaptamos para salvar o melhor genoma dentre todos em uma população com base naquele que possui o melhor valor de fitness.

### 3. Arquivo config-feedforward

Este arquivo é o responsável por definir os parâmetros do modelo. A cada vez que o arquivo é executado no modo "iniciar novo treino" ou "continuar treino" as configurações presentes nele são utilizadas para definir parâmetros importantes como a definição do valor máximo de fitness, a função de ativação que será utilizada , taxa de extinção da população, entre outros. Um exemplo simples utilizando uma função XOR que utiliza um arquivo feedforward pode ser consultado [aqui](https://neat-python.readthedocs.io/en/latest/xor_example.html).


## Referências

**1. Vídeo do YouTube:**

- THOMPSON, Lucas. Results of Tutorial on Sonic 2 - EP0 - Open-AI and NEAT Tutorial. YouTube, 3 de agosto de 2018. Disponível em: https://www.youtube.com/watch?v=pClGmU1JEsM&list=PLTWFMbPFsvz3CeozHfeuJIXWAJMkPtAdS&index=1&t=0s&ab_channel=LucasThompson. Acesso em: 5 de dezembro de 2023.

**2. Repositório GitLab:**

- THOMPSON, Lucas. Sonic Bot in OpenAI and NEAT. GitLab, 2021. Disponível em: https://gitlab.com/lucasrthompson/Sonic-Bot-In-OpenAI-and-NEAT. Acesso em: 5 de dezembro de 2023.

**3. Código-fonte no GitHub:**

- CODERECLAIMERS. neat-python - checkpoint.py. GitHub, 2022. Disponível em: https://github.com/CodeReclaimers/neat-python/blob/master/neat/checkpoint.py. Acesso em: 5 de dezembro de 2023.

**4. Repositório no GitHub:**

- UFABC-BCC. Super Mario World - pbacellar. GitHub, 20 de agosto de 2021. Disponível em: https://github.com/ufabc-bcc/super-mario-world-pbacellar. Acesso em: 5 de dezembro de 2023.

