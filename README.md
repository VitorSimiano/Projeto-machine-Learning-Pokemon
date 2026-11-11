# Reconhecimento de Pokémon em Vídeo (MobileNetV2)

## Estrutura da pasta

```
pokemon_video/
  utils.py  extrair_frames.py  train.py  detectar.py
  videos/
    pikachu/  psyduck/  charmander/  fundo/     <- você cria e coloca os vídeos
  (gerados) data/  modelo.keras  class_names.json  curvas.png
            matriz_fase1.png  matriz_fase2.png  dataset_imagens.pdf
```

Instalar: `pip install tensorflow numpy matplotlib opencv-python pillow`
(sem GPU: use o Google Colab só para o `train.py`; o `detectar.py` roda bem na CPU).

## 1. Gravação (o que o PDF exige)

- Mínimo de 3 Pokémon (sugestão: Pikachu + Psyduck, que são amarelos, e Charmander) + classe `fundo`.
- Mínimo de 4 vídeos por classe. **Grave 6** por Pokémon: com 4 vídeos só 1 vai para validação e 3 para treino; com 6, são 4 treino / 1-2 validação.
- Metade em fundo claro, metade em fundo escuro; todo Pokémon em todos os cenários (senão o modelo aprende o cenário).
- Cada vídeo mistura ângulos (frente, costas, lados, cima, baixo, perto e longe), 10 a 30 s.
- Para a Pergunta 3, grave **1 vídeo por Pokémon só de costas** e coloque `costas` no nome (ex.: `v05_costas_claro.mp4`). Os outros vídeos não devem ter essa palavra no nome.
- `fundo`: mesa vazia, parede, mão sem Pokémon, outros objetos, nos dois cenários.
- Nomes sem `__` (dois underlines): `v01_frente_claro.mp4`.
- Vídeo de teste (30 a 60 s): outro lugar, nunca usado no treino, os 3 Pokémon um de cada vez. Guarde fora de `videos/`.

## 2. Rodar

```
python extrair_frames.py
python train.py                      # split por vídeo + fine-tuning (gera PDF, curvas, matrizes)
python detectar.py video_teste.mp4   # janela 8, limiar 60%
```

Testar janelas: `--janela 1`, `--janela 8`, `--janela 15`. Testar limiar: `--limiar 0.5 / 0.6 / 0.8`.

## 3. Experimentos da Parte 5

| Pergunta | Comandos |
|---|---|
| 1. Split por vídeo x por foto | `python train.py --split foto --tag split_foto` e depois `python detectar.py video_teste.mp4 --modelo modelo_split_foto.keras --sufixo _foto` (compare com o modelo normal) |
| 2. Fine-tuning ajudou? | Compare `matriz_fase1.png` x `matriz_fase2.png` e as curvas. Depois: `python train.py --lr-ft 1e-3 --sem-early --tag ft_1e3` e veja a val_loss em `curvas_ft_1e3.png` |
| 3. Ângulo novo | `python train.py --excluir pikachu:costas --tag sem_costas` e rode o detectar num vídeo do Pikachu de costas com `--modelo modelo_sem_costas.keras` |
| 4. Pokémon desconhecido | Grave um 4º Pokémon (ex.: Bulbasaur) e rode `detectar.py` nele. Anote o rótulo e a % mostrada |
| 5. Dois no quadro | Grave um trecho com 2 Pokémon juntos e rode `detectar.py`. Anote o que apareceu |

## 4. Entrega

1. `dataset_imagens.pdf`
2. `saida_<nome>.mp4` do vídeo de teste + print do terminal com a linha do tempo
3. `train.py` e `detectar.py` (já comentados nas partes das melhorias)
4. `curvas.png` + print da matriz de confusão
5. Relatório em PDF com as respostas da Parte 5 (use `relatorio_modelo.md` como roteiro)

Só são aceitos trabalhos com os itens 1 e 2.
