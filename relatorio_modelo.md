# Relatório - Reconhecimento de Pokémon em Vídeo

**Grupo:** (nomes)  **Pokémon escolhidos:** (3 ou mais) + fundo

## Dados
Quantidade de vídeos e de fotos por classe, cenários usados e como os ângulos foram cobertos. Cole 1 print do `extrair_frames.py`.

## Resultados principais
Acurácia de validação (fase 1 e fase 2), matriz de confusão, resultado da linha do tempo no vídeo de teste, janela e limiar escolhidos e por quê.

## Pergunta 1 - Por que separar por vídeo?
Experimento: treino com split por vídeo x split por foto. Tabela com a acurácia de validação dos dois e o resultado de cada um no vídeo de teste. Qual número representa o desempenho real? Explicar com o conceito de vazamento de dados (frames vizinhos quase idênticos).

## Pergunta 2 - O fine-tuning ajudou?
Comparar fase 1 x fase 2 (curvas e matriz). Repetir com lr 1e-3: o que houve com a val_loss? Por que a taxa precisa ser tão baixa (pesos pré-treinados, gradientes grandes destroem os filtros)?

## Pergunta 3 - Qualquer ângulo?
Pokémon e ângulo excluídos, o que o modelo previu no vídeo de costas e com qual confiança, quais características ele parece usar (cor, formato, orelhas, cauda) e duas estratégias de robustez (ex.: mais vídeos com ângulos variados, augmentation mais forte, mais camadas no fine-tuning).

## Pergunta 4 - Pokémon desconhecido
Qual Pokémon, o que o modelo mostrou (? ou um dos três) e a confiança. Explicar o softmax (soma 100% entre as classes conhecidas) e como o limiar e a classe fundo ajudam ou não.

## Pergunta 5 - Classificação ou detecção?
O que aconteceu com dois Pokémon no quadro. Diferença entre classificação e detecção, anotação necessária (caixas delimitadoras por objeto) e modelo (ex.: YOLO).

## Anexos
Prints, gráficos e trechos do vídeo.
