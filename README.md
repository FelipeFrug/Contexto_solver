# Contexto_solver

Feito por: Felipe Frug & Albert Hamoui

Solver automático para o jogo [Contexto](https://contexto.me/en/), desenvolvido com técnicas de Processamento de Linguagem Natural.

## Como funciona

O programa usa embeddings do modelo **BERT** (`bert-base-uncased`) para medir similaridade semântica entre palavras. A cada rodada, ele calcula um vetor médio ponderado das melhores tentativas conhecidas e busca os candidatos mais próximos nesse espaço semântico.

**Fluxo geral:**
1. Na primeira execução, pré-computa embeddings para ~18k palavras do vocabulário BERT e salva em `embeddingsValues.npz` (nas execuções seguintes carrega do arquivo).
2. Começa explorando temas genéricos ("food", "profession", "animal"…) para abrir diferentes regiões semânticas.
3. A partir das melhores palavras encontradas, calcula um vetor médio (peso = `1/(score+1)`, então scores baixos têm mais peso) e sugere os candidatos mais similares.
4. Filtra variantes morfológicas (reach/reaches/reaching) usando **Porter Stemmer** — uma vez que uma raiz é testada, todas as suas variações são bloqueadas.
5. Se não houver melhora nos melhores resultados, expande automaticamente o número de sugestões por rodada.
6. A automação no navegador é feita via **Selenium**, que digita as palavras e lê os scores diretamente do site.

**Classificação por score:**
| Faixa | Categoria |
|-------|-----------|
| < 50  | perfeitos |
| 50–199 | ótimos |
| 200–299 | bons |
| 300–499 | ok |
| ≥ 500 | ruins |

## Tecnologias

- Python / Jupyter Notebook
- [HuggingFace Transformers](https://huggingface.co/) — BERT (`bert-base-uncased`)
- Selenium + WebDriver Manager — automação do navegador
- NumPy + scikit-learn — operações vetoriais e similaridade de cosseno
- NLTK (Porter Stemmer) — filtragem de variantes morfológicas

## Arquivos

- `main.ipynb` — implementação completa
- `embeddingsValues.npz` — embeddings pré-computados (gerado automaticamente na primeira execução)

## Como executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/FelipeFrug/Contexto_solver.git
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Abra `main.ipynb` e execute as células em ordem. O Chrome abrirá automaticamente com o jogo em andamento.

> Na primeira execução o pré-cômputo dos embeddings pode levar alguns minutos. Nas seguintes é instantâneo.

## Vídeo

https://www.youtube.com/watch?v=sW2Y1J0ncck
