##------------------------------------------------------------------------------------##
## Import das biblioetecas
##------------------------------------------------------------------------------------##
import retro
from rominfo import*
import neat
##---------------------------------------------------------------------------------##
import os
import pickle
import numpy as np
import argparse
##------------------------------------------------------------------------------------##
# Passagem de argumentos para o main
##------------------------------------------------------------------------------------##
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rodando um modelo')
    parser.add_argument('modelo',type=str, help="Path para o modelo no formato ./path/modelo.pkl ")
    parser.add_argument('--fase', type=str, default='YoshiIsland2', help="Fase do Super Mario World")
    
    args = parser.parse_args()
##------------------------------------------------------------------------------------##
    ## Inicializa o ambiente
##------------------------------------------------------------------------------------##
    env = retro.make(game="SuperMarioWorld-Snes", state=args.fase, players=1)
    env.reset()

    configuracaoNeat = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward')

    with open(args.modelo, 'rb') as f:
        modelo = pickle.load(f)
        
    recurrentNetwork = neat.nn.recurrent.RecurrentNetwork.create(modelo, configuracaoNeat)
    
    done = False
    
    while not done:
        env.render()
        obs, x, y = getInputs(getRam(env))
        outPut = np.round(recurrentNetwork.activate(obs))
        ob, reward, done, info = env.step(outPut)
        
        
     
        