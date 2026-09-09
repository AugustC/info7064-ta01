# INFO7064 — TA01 (Análise de Textura)

Este repositório contém os *scripts* e o *notebook* usados para reproduzir a análise descrita no relatório (arquivo `TA01___Texture_Analysis.pdf`).

## Estrutura do repositório

- `Analysis.ipynb`: notebook principal com o fluxo completo de análise.
- `src/`: código-fonte em Python.
  - `filters.py`: filtros e pré-processamento.
  - `image_patch.py`: extração/manipulação de *patches* de imagem.
  - `classification.py`: rotinas de classificação.
- `outputs/`: imagens/figuras geradas (ex.: `original_img.png`, `segmentation.png`, etc.).
- `pyproject.toml` / `uv.lock`: definição e *lock* de dependências.

## Requisitos

- Python $\ge$ 3.10.
- Dependências listadas em `pyproject.toml` (ex.: Jupyter, Matplotlib, OpenCV).

## Como executar

```bash
uv sync
uv run jupyter lab
```

Abra o notebook `Analysis.ipynb` e execute as células na ordem.

## Reprodutibilidade

- Caso você altere parâmetros no notebook, os resultados podem mudar (o que é esperado).

## Autoria

Trabalho desenvolvido para a disciplina INFO7064 (TA01) - Visão Computacional e Percepção por Augusto César Monteiro Silva
