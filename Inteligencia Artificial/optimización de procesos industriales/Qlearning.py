# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 22:07:37 2020

@author: Efrain Santos Luna

Email: esl_elect@outlook.com

cellphone: 55-66-18-62-95
"""

import numpy as np

def Qlearning(location_to_state,gamma ,alpha ,ending_location = 'G'):

    '''
    intermediate_location: Localizacion intermedia
    gamma: factor de descuento por defecto 0.9
    alpha: que tan rapido va a converger pero tambien que tanto va a aprender  por defecto 0.75
    ending_location: posicion final a la que se quiere llegar
    '''
    
    #definicion de las recompensas (matris) revisar el mapa del entorno en las diapositivas
    
    #Columnas      A B C D E F G H I J K L
    R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],  #A
    			  [1,0,1,0,0,1,0,0,0,0,0,0],  #B
    			  [0,1,0,0,0,0,1,0,0,0,0,0],  #C
    			  [0,0,0,0,0,0,0,1,0,0,0,0],  #D
    			  [0,0,0,0,0,0,0,0,1,0,0,0],  #E
    			  [0,1,0,0,0,0,0,0,0,1,0,0],  #F
    			  [0,0,1,0,0,0,1,1,0,0,0,0],  #G
    			  [0,0,0,1,0,0,1,0,0,0,0,1],  #H
    			  [0,0,0,0,1,0,0,0,0,1,0,0],  #I
    			  [0,0,0,0,0,1,0,0,1,0,1,0],  #J
    			  [0,0,0,0,0,0,0,0,0,1,0,1],  #K
    			  [0,0,0,0,0,0,0,1,0,0,1,0]]) #L
    
    ending_state = location_to_state[ending_location]
    R[ending_state,ending_state] = 1000
    
    #PARTE 2 - CONSTRUCCION DE LA SOLUCION DE IA
    
    #Inicializacion de los Qvalues
    Q = np.array(np.zeros([12,12]))
    
    #implementacion del proceso de Qlearning
    
    for i in range(1000):
        
        #paso 1, seleccionar un estado aleatorio S_t de los 12 posible
        current_state = np.random.randint(0,12)
        
        #paso 2-3, realizar una accion que lleve a una recompenza positiva para conducir al siguiente estado posible
        playable_actions =[]
        for j in range(12):
            reward = R[current_state,j]
            if(reward > 0):
                playable_actions.append(j)
        #al guardar la accion aleatoria a_t estamos haciendo el paso 3 ya que esa accion coincide para este caso con el siguiente estado
        next_state = np.random.choice(playable_actions)
        #accion actual a_t
        current_action = next_state
        
        #Paso 4 calcular la diferencia temporal
        TD = R[current_state, current_action] + gamma*Q[next_state, np.argmax(Q[next_state,])] - Q[current_state,current_action]
        
        #Paso 5 actualizar el valor de Q en base a la escuacion de Bellman
        Q[current_state, current_action] = Q[current_state, current_action] + alpha*TD
    
    return Q.astype(int)
    
#PARTE 3 PONER EL MODELO EN PRODUCCION

#funcino final que devuelva la ruta optima
def routine(location_to_state,state_to_location,Q,starting_location = 'E', ending_location='G'):
    '''
    Parameters
    ----------
    location_to_state : diccionario
        es el diccionario donde las localizacion son las keys.
    state_to_location : diccionario
        es el diccionario donde los estados son las keys.
    Q : np.array
        es la matriz de Qvalues (recompensas) derivada del entrenamiento.
    starting_location : string, optional
        posicion incial de donde parte el robot. The default is 'E'.
    ending_location : strin, optional
        posicion final a la que se quiere llegar. The default is 'G'.

    Returns
    -------
    route : lista
        Lista con la ruta elegida por el robot.
    '''
    
    #starting location es la letra y starting_state es el numero del diccionario de estados
    route = [starting_location]
    #solo para inicializar
    next_location = starting_location
    while(next_location != ending_location):
        starting_state = location_to_state[starting_location]
        #priorizamos segun las localizaciones
        aux_state = np.argmax(Q[starting_state,])
        #verificamos si hay dos stados con el mismo valor de recompenza
        indices = np.where(Q[starting_state,]==aux_state)
        if (len(indices) > 1):
            state_priorities = []
            for x in indices:
                state_priorities.append(location_to_state.values()[x])
            next_state = max(state_priorities)
        else:
            next_state = aux_state
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
        
    return route



def start_boot(start_location = 'E',intermediate_location = '',end_location = 'G',gamma = 0.75,alpha = 0.9):
    '''
    
    Parameters
    ----------
    start_location : TYPE, string
        DESCRIPTION. Es la localizacion incial.The default is 'E'.
    intermediate_location : TYPE, string 
        DESCRIPTION. es la localizacion intermedia a donde se quiere llegar. The default is ''.
    end_location : TYPE, string
        DESCRIPTION. es la localizacion final a donde se requiere llegar.The default is 'G'.
    gamma : TYPE, float
        DESCRIPTION. factor de descuento por defecto 0.9
    alpha : TYPE, float
        DESCRIPTION. que tan rapido va a converger el algoritmo pero tambien que tanto va a aprender  por defecto 0.75

    Returns
    -------
    None.

    '''
    
    #PARTE 1 - DEFINICION DEL ENTORNO
    
    #Definicion de los estados
    location_to_state = {'A': 0,
    					 'B': 1,
    					 'C': 2,
    					 'D': 3,
    					 'E': 4,
    					 'F': 5,
    					 'G': 6,
    					 'H': 7,
    					 'I': 8,
    					 'J': 9,
    					 'K': 10,
    					 'L': 11}
    
    #tansformacion inversa de estados a ubicaciones

    state_to_location = dict(zip(location_to_state.values(),location_to_state.keys()))
    
    #Si existe una ubicacion intermedia se corre dos veces al robot
    if(intermediate_location != ''):
        #entrenamos el robot con el metodo de Qlearning de la ubicacion incial a la intermedia
        Q = Qlearning(ending_location=intermediate_location,location_to_state = location_to_state,gamma = gamma,alpha = alpha)
        ruta_1 = routine(starting_location=start_location,ending_location=intermediate_location,location_to_state=location_to_state,state_to_location=state_to_location,Q=Q)
        
        #entrenamos el robot de la ubicacion intermedia a la final
        Q = Qlearning(ending_location=end_location,location_to_state = location_to_state,gamma = gamma,alpha = alpha)
        ruta_2 = routine(starting_location=intermediate_location,ending_location=end_location,location_to_state=location_to_state,state_to_location=state_to_location,Q=Q)
        
        ruta_final = ruta_1 + ruta_2[1:]
    else:
        #entrenamos el robot con el metodo de Qlearning de la ubicacion incial a la final
        Q = Qlearning(ending_location=end_location,location_to_state = location_to_state,gamma = gamma,alpha = alpha)
        ruta_final = routine(starting_location=start_location,ending_location=end_location,location_to_state=location_to_state,state_to_location=state_to_location,Q=Q)
        
    print('ruta eleginda:')
    print(ruta_final)

    return



'''
con la llamada a la funcion start_boot se realiza todo el entrenamiento del robot y la busqueda de la mejor ruta
se pueden modificar los parametros de alpha, gamma, localizacion incial, intermedia y final, todos los parametros
son opcionales, a continuacion se describe cada parametro:
    
Parameters
    ----------
    start_location : TYPE, string (Upper)
        DESCRIPTION. Es la localizacion incial.The default is 'E'.
    intermediate_location : TYPE, string (Upper)
        DESCRIPTION. es la localizacion intermedia a donde se quiere llegar. The default is ''.
    end_location : TYPE, string (Upper)
        DESCRIPTION. es la localizacion final a donde se requiere llegar.The default is 'G'.
    gamma : TYPE, float
        DESCRIPTION. factor de descuento por defecto 0.9
    alpha : TYPE, float
        DESCRIPTION. que tan rapido va a converger el algoritmo pero tambien que tanto va a aprender  por defecto 0.75

'''

start_boot()

        








