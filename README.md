# Contexto_solver

Feito por:
Felipe Frug & Albert Hamoui

Este projeto tem como objetivo desenvolver um programa capaz de resolver o jogo [Contexto](https://contexto.me) utilizando conhecimentos adquiridos no curso de Processamento de Linguagem Natural.

## Sobre o Projeto

O Contexto_solver aplica técnicas de PLN para identificar a palavra secreta do jogo Contexto.  
Utilizando embeddings de palavras e algoritmos de similaridade semântica, o programa realiza tentativas iterativas até encontrar a palavra correta.

## Tecnologias Utilizadas

- Python  
- Jupyter Notebook  
- NumPy  
- Modelos de Embeddings (ex: BERT)

## Estrutura do Projeto

O repositório contém os seguintes arquivos e pastas:

- `novo_main.ipynb`: Notebook principal com a implementação do algoritmo.
- `requirements.txt`: Lista de dependências necessárias para executar o projeto.

## Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/FelipeFrug/Contexto_solver.git
   ```
2. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
3. Abra o notebook novo_main.ipynb e execute as células para iniciar o processo de resolução do jogo.

## Decisões feitas

- O codigo pega as 10 palavras mais próximas de uma vez para ficar um pouco mais rápido.
- Ao obter uma palavra no top 10 o codigo está sendo considerado bem sucedido. 
- O codigo pega grupos de palavras aleatórias e desse grupo escolhe as melhores, por isso pode haver um aspecto um pouco aleatório no código. Isso foi feito pois quando feito com todas as palvras demora muito para processar.

## Video

https://www.youtube.com/watch?v=sW2Y1J0ncck
