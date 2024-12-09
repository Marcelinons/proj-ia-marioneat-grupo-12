##----------------------------------------------------------------------------------------------------------##
## Usando neuroevolução para treinar um algoritivo NEAT para jogar Super Mario World, na fase Yoshiland2
##----------------------------------------------------------------------------------------------------------##

##----------------------------------------------------------------------------------------------------------##
##Import das bliotecas utilizadas
##----------------------------------------------------------------------------------------------------------##

##Import da biblioteca Neat --> https://neat-python.readthedocs.io/en/latest/neat_overview.html

import neat.genome
import numpy as np
import os
import argparse
import retro
import neat
import random
import pickle
import gzip
from rominfo import *
import time

##------------------------------------------------------------------------------------##
## Algumas anotações sobre a biblioteca Retro
##------------------------------------------------------------------------------------##

#Below is a list of the interesting buttons and their representation as an array of bits
#jump          [1,0,0,0,0,0,0,0,0,0,0,0]
#run modifier  [0,1,0,0,0,0,0,0,0,0,0,0] (there is another button for running, but only one is needed)
#down          [0,0,0,0,0,1,0,0,0,0,0,0]
#left          [0,0,0,0,0,0,1,0,0,0,0,0]
#right         [0,0,0,0,0,0,0,1,0,0,0,0]
#spin jump     [0,0,0,0,0,0,0,0,1,0,0,0]

#Funções
# while not done                            --> enquanto não perdermos
# env.step(action)                          --> 
# env.action_space.sample() --> ação aleatória.
# emp.render() --> a exibição gráfica do estado atual do jogo ou ambiente.
# x, y, c = env.observation_space.shape() --> x, y e numero de cores do ambiente de observação
# ob, reward, done, info = env.step(pred)
# ob -- Image of the screen at the time of the action
# rw -- amount of reward that 
# done -- done condition
# info -- dictionary of all of the values uou've set in data


##------------------------------------------------------------------------------------##
## Inicializa o ambiente
##------------------------------------------------------------------------------------##

env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2', players=1) #cria o ambiente
env.reset()

##------------------------------------------------------------------------------------##
## Configuração da biblioteca neat. Arquivo retirado da documentação, e parâmetros alterados
##------------------------------------------------------------------------------------##
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward')


def calculate_genome_fitness(genomes:neat.genome, configuracaoNeat:neat.Config): 
    """Atualiza a aptidao (fitness) de cada genoma na populacao a partir
    da "distancia percorrida" pelo player na fase.

    Args:
        genomes (neat.genome): Iterable contendo as tupas (genome_id, genome)
        configuracaoNeat (neat.Config): Configuracao do NEAT
    """
    for genome_id, genome in genomes:
        env.reset()             # Volta o jogo para o estado inicial 

        # Caracteristicas iniciais do jogo
        obs, x, y = getInputs(getRam(env))
        xInicial = x
        
        # Rede neural com as informações do geneticas do genoma
        recurrentNetwork = neat.nn.recurrent.RecurrentNetwork.create(genome,configuracaoNeat)
        
        genome.fitness = _calculate_fitness(genome_id, recurrentNetwork, xInicial, obs)  
        
        
def _calculate_fitness(genome_id, network, xInicial, obs) -> int:
    """ Executa uma run do jogo para o genome_id 
    
    Returns:
        int: fitness do genoma 
    """
    done = False    # Fim de jogo, seja vitoria ou derrota
    _i = 0          # Contador de iteracoes para verificar se o cronometro congelou
    _j = 500        # Variavel temporaria para salvar o valor anterior do cronometro
    fitness = 0
    recompensa = 0
    curr_stpd_time = None
    while not done:
        # Enquanto Mario nao morrer ou nao completar a fase
        _i += 1

        env.render()
        obs, x, y = getInputs(getRam(env))
        outPut = np.round(network.activate(obs))   # Ativo a rede neural e prevejo a saída(ação)
        ob, reward, done, info = env.step(outPut)           # Aplica a ação
        
        # Calcula a distancia percorrida com a acao tomada e atualiza a variavel de aptidao
        distanciaPercorrida = x - xInicial
        fitness += int(distanciaPercorrida)
        
        recompensa += reward
        
        xInicial = x # Agora o xInicial passa a ser o X que chegamos com a ação tomada
        
        memoriaRam = env.get_ram()
        
        #Acessa a memória 0x71 que de acordo com o jogo contém informações sobre o estado em que perdemos o jogo
        gameOver = memoriaRam[0X71] 
        
        # Acessa novamente a memória RAM para o tempo em segundos do cronometro da fase
        curr_lvl_time = (memoriaRam[0x0F31] * 100) + (memoriaRam[0x0F32] * 10) + memoriaRam[0x0F33] 
        
        # Se Mario ficou preso em um pop up de tutorial, check a cada 120 iteracoes
        if _i % 120 == 0:
            if curr_lvl_time == _j:
                print('O genoma {} ficou preso em um popup tutorial.    Fitness alcançado: {}'.format(genome_id, fitness))
                return fitness 
            else: _j = curr_lvl_time
        
        # Se Mario demorou demais
        if curr_lvl_time < 270:
            print('O genoma {} não completou a fase em tempo bom.   Fitness alcançado: {}'.format(genome_id, fitness))
            return fitness
        
        # Se o mario chegou na posição final
        if x >= 4809: #Posição final do X
            fitness = 5000 + recompensa/1000 #5000 #valor atribuido no arquivo de configuração
            print('O genoma {} chegou ao final da fase.             Fitness alcancado: {}'.format(genome_id, fitness))
            return fitness
            
        # Se o mario morreu
        if gameOver == 9:
            print('O genoma {} morreu.                              Fitness alcancado: {}'.format(genome_id, fitness))
            return fitness
            
        # Se o mario ficou parado, marca o momento
        if distanciaPercorrida == 0 and curr_stpd_time == None:    
            curr_stpd_time = (memoriaRam[0x0F31] * 100) + (memoriaRam[0x0F32] * 10) + memoriaRam[0x0F33]
        
        # Reseta o marcador do tempo parado quando Mario se move novamente
        if distanciaPercorrida != 0:
            curr_stpd_time = None
        
        #Se o mario ficou parado por 4 segundos ou mais
        if (curr_stpd_time != None and (distanciaPercorrida == 0 and (curr_stpd_time - curr_lvl_time) >= 4)):
            print('O genoma {} ficou parado.                        Fitness alcancado: {}'.format(genome_id, fitness))
            return fitness
        

##------------------------------------------------------------------------------------##
## Função para criar novo treinamento
##------------------------------------------------------------------------------------##
def novoTreinamento(diretorioTreinamento:str, freqSalvamento:int=1) -> None:
    """ Inicia um novo treinamento
    """
    configuracaoNeat = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward')
    population = neat.Population(configuracaoNeat)
    
    #Cria a pasta para salvar o checkpoint
    flPastaCriada = 0
    while flPastaCriada == 0:
        path = './' + diretorioTreinamento
        if not os.path.exists(path):
            os.makedirs(path)
            flPastaCriada = 1
        else:
            diretorioTreinamento = str(input("Forneca o nome de outra pasta que nao exista-->"))
    
    #Adiciona estatísticas (avg da população, desvio padrão e melhor fitness)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
   
    #Checkpointer
    population.add_reporter(MeuChekpointer(generation_interval = freqSalvamento, filename_prefix=('./'+path+'/neat-checkpoint-generation-')))
    
    #Jogador
    player = population.run(calculate_genome_fitness)
    
    #Salva o melhor modelo após o término do treinamento
    print("Salvando o modelo campeão")
    with open('TheBestModel.pkl', 'wb') as modeloTreinado:
        pickle.dump(player, modeloTreinado, 1)
        modeloTreinado.close()
    
    #Crio a rede neural com as informações do genéticas do melhor genoma/modelo
    recurrentNetwork = neat.nn.recurrent.RecurrentNetwork.create(player,configuracaoNeat)
    
    
##------------------------------------------------------------------------------------##
## Função para continuar um treinamento
##------------------------------------------------------------------------------------##
def continuarTreinamento(diretorioTreinamento:str, arquivoCheckpoint:str, freqSalvamento:int=1) -> None:
    """ Continua um treinamento a partir do arquivo de checkpoint
    """    
    configuracaoNeat = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward')

    path = './' + diretorioTreinamento + '/' + arquivoCheckpoint
    
    #Cria a população a partir do arquivo de checkpoint
    population = neat.Checkpointer.restore_checkpoint(path)
    
    #Adiciona estatísticas (avg da população, desvio padrão e melhor fitness)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
   
    #Checkpointer
    population.add_reporter(MeuChekpointer(generation_interval=freqSalvamento, 
                                           filename_prefix=('./'+diretorioTreinamento+'/neat-checkpoint-generation-')))
    
    # Executa o treinamento para a populacao
    player = population.run(calculate_genome_fitness)
    
    #Salva o melhor modelo após o término do treinamento
    print("Salvando o modelo campeão")
    with open('TheBestModel.pkl', 'wb') as modeloTreinado:
        pickle.dump(player, modeloTreinado, 1)
        modeloTreinado.close()
    
    #Crio a rede neural com as informações do genéticas do melhor genoma/modelo
    recurrentNetwork = neat.nn.recurrent.RecurrentNetwork.create(player,configuracaoNeat)
    
##------------------------------------------------------------------------------------##
## Classe para salvar o treinamento (usada nas funções abaixo)
## Retirado de: https://github.com/CodeReclaimers/neat-python/blob/master/neat/checkpoint.py
##------------------------------------------------------------------------------------##
class MeuChekpointer(neat.Checkpointer):
  
    def __init__(self, generation_interval=10, time_interval_seconds=300, filename_prefix='neat-checkpoint-'):
        self.generation_interval = generation_interval
        self.time_interval_seconds = time_interval_seconds
        self.filename_prefix = filename_prefix

        self.current_generation = None
        self.last_generation_checkpoint = -1
        self.last_time_checkpoint = time.time()

    def save_checkpoint(self, config, population, species_set, generation):
        """ Save the current simulation state. """
        filename = '{0}{1}'.format(self.filename_prefix, generation)
        print('Saving checkpoint to {0}'.format(filename))

        with gzip.open(filename, 'w', compresslevel=5) as f:
            data = (generation, config, population, species_set, random.getstate())
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        """Salva o melhor genoma da geracao"""
        #import ipdb; ipdb.set_trace()
        MelhorGenoma = None
        for genoma in population:
            if population[genoma].fitness != None:
                if MelhorGenoma is None or population[genoma].fitness > MelhorGenoma.fitness:
                    MelhorGenoma = population[genoma]

        MelhorModelo = MelhorGenoma
        modeloName = '{0}{1}{2}'.format(self.filename_prefix, '- melhor genoma generation - ',generation) #generation)
        modeloName = modeloName  + '.pkl'
        print('Saving best fit model checkpoint to {0}'.format(modeloName ))

        with open(modeloName , 'wb') as output:
            pickle.dump(MelhorModelo, output, 1) # salva o melhor genoma da geração
            output.close()
##------------------------------------------------------------------------------------##
## Função main
##------------------------------------------------------------------------------------##

def main():
    parser = argparse.ArgumentParser(description='Treinamento da IA')
    parser.add_argument('modo_treino', choices=['novo', 'continuar'], help='Modo de treinamento: novo ou continuar')
    parser.add_argument('diretorio', help='Caminho para o diretório de treinamento')
    #parser.add_argument('--nome_arquivo', help='Nome do arquivo de treinamento (opcional, apenas para continuar)')
    parser.add_argument('arquivo_chekpoint', type=str, default='None', help="Nome do arquivo de treinamento (opcional, apenas para continuar)")
    args = parser.parse_args()
    
    if args.modo_treino == 'novo':
        novoTreinamento(args.diretorio, 1)
    elif args.modo_treino == 'continuar':
        continuarTreinamento(args.diretorio,args.arquivo_chekpoint, 1)


        
if __name__ == "__main__":
    main()



