#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:17:48 2023

@author: NullCipherr
"""

import numpy as np
import cv2
import os

depuration_folder = "Depuration"



##############################################################################
# Função 'carregar_imagem' : Carrega uma imagem qualquer.
##############################################################################
# Parâmetros
#   - caminho_entrada : Caminho ao qual será localizado a imagem desejada.
##############################################################################
def carregar_imagem(caminho_entrada) :
    try:
        imagem = cv2.imread(caminho_entrada)
        
        if imagem is not None :
            return imagem
        else:
            print("Erro ao carregar a imagem no caminho especificado ... ")
            return None
    except Exception as e:
            print(f"Erro ao carregar a imagem: {e}")
            return None
      
        
      
        
##############################################################################
# Função 'salvar_imagem' : Salva uma imagem qualquer em um caminho especifico.
##############################################################################
# Parâmetros
#   - image: Imagem passada por parametro para ser salva no local desejado.
#   - caminho_saida : Caminho que a imagem será salva.
##############################################################################
def salvar_imagem(image, nome_saida, pasta_saida) :
    
    print("\nNome de saida -> ",  nome_saida, "\nPasta de saida -> ", pasta_saida)  
    
    # Cria o caminho completo para a localização
    caminho_saida = os.path.join(pasta_saida, nome_saida)
    
    try:
        cv2.imwrite(caminho_saida, image)
        print(f"Imagem salva com sucesso em {caminho_saida}")
    except Exception as e:
        print(f"Erro ao salvar a imagem: {e}")
        
        
        
        
##############################################################################
# Função 'calcular_normal_map' : Calcula o normal map de uma imagem.
##############################################################################
# Parâmetros
#   - image : Imagem desejada para calcular o normal map.
##############################################################################
def calcular_normal_map(image):
    
    
    # Converte a imagem para escala de cinza (necessário para o cálculo do gradiente)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Salvar a imagem em escala de cinza para depuração
    salvar_imagem(gray_image, "Gray.jpg", depuration_folder)

    # Calcula o gradiente horizontal e vertical usando o operador Sobel
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    
    # Salvar as imagens de gradientes para depuração
    salvar_imagem(gradient_x, "Gradient_x.jpg", depuration_folder)
    salvar_imagem(gradient_y, "Gradient_y.jpg", depuration_folder)

    # Normaliza os gradientes para o intervalo [-1, 1]
    gradient_x = cv2.normalize(gradient_x, None, -1, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    gradient_y = cv2.normalize(gradient_y, None, -1, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    
    normalized_gradient_x = (gradient_x * 255).astype(np.uint8)
    normalized_gradient_y = (gradient_y * 255).astype(np.uint8)
    
    # Salvar as imagens de gradientes para depuração
    salvar_imagem(normalized_gradient_x, "Normalized_Gradient_x.jpg", depuration_folder)
    salvar_imagem(normalized_gradient_y, "Normalized_Gradient_y.jpg", depuration_folder)

    # cv2.imwrite("normalized_gradient_x.jpg", (gradient_x * 255).astype(np.uint8))
    # cv2.imwrite("normalized_gradient_y.jpg", (gradient_y * 255).astype(np.uint8))

    # Calcula a componente Z da normal usando a fórmula de normalização
    component_z = np.sqrt(np.clip(1.0 - gradient_x**2 - gradient_y**2, 0, 1))
    
    normal_z = (component_z * 255).astype(np.uint8)

    # Salvar a imagem de normal_z para depuração
    salvar_imagem(normal_z, "Normal_z.jpg", depuration_folder)
    

    # Empilha as componentes da normal em uma imagem RGB
    normal_map = np.dstack((gradient_x, gradient_y, component_z))

    # Salvar a imagem do normal map para depuração
    # cv2.imwrite("normal_map.jpg", (normal_map * 128 + 128).astype(np.uint8))

    return (normal_map * 128 + 128).astype(np.uint8)
        
        
        
        
        
if __name__ == "__main__":
    
    # Caminho da imagem de entrada.
    imagem_entrada = "Texture.jpg"
    
    # Caminho para a pasta de entrada.
    pasta_entrada = "Input"
    
    # Carrega a imagem de entrada.
    caminho_imagem_entrada = os.path.join(pasta_entrada, imagem_entrada)
    
    # Carrega a imagem de entrada.
    imagem_entrada = carregar_imagem(caminho_imagem_entrada)

    if imagem_entrada is not None:
        # Aplica a função 'calcular_normal_map' na imagem de entrada..
        normalMap = calcular_normal_map(imagem_entrada)

        # Salva a imagem de entrada.
        salvar_imagem(normalMap, "Normal_Map.jpg", "Output")