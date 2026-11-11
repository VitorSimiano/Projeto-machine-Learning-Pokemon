"""
Passo 3 - Reconhece Pokemon em um video (ou na webcam), com:
  MELHORIA 1: suavizacao (media dos ultimos N frames) + limiar de confianca ("?")
  MELHORIA 2: linha do tempo impressa no final

Uso:
  python detectar.py video_teste.mp4                 -> gera saida_video_teste.mp4
  python detectar.py video_teste.mp4 --janela 15 --limiar 0.7
  python detectar.py video_teste.mp4 --modelo modelo_split_foto.keras --sufixo _foto
  python detectar.py 0                               -> webcam (q = sair)
  python detectar.py video_teste.mp4 --sem-janela    -> processa sem abrir janela (mais rapido)
"""
import os
import json
import argparse
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from utils import letterbox

ap = argparse.ArgumentParser()
ap.add_argument("fonte", nargs="?", default="0", help="caminho do video ou 0 para webcam")
ap.add_argument("--modelo", default="modelo.keras")
ap.add_argument("--janela", type=int, default=8, help="N frames da media movel (1 = sem suavizacao)")
ap.add_argument("--limiar", type=float, default=0.60, help="abaixo disso mostra '?'")
ap.add_argument("--min-trecho", type=float, default=0.5, help="trechos menores (s) sao ignorados")
ap.add_argument("--lacuna", type=float, default=0.3,
                help="buraco (s) entre dois trechos do mesmo Pokemon que e fundido")
ap.add_argument("--sufixo", default="", help="sufixo dos arquivos de saida")
ap.add_argument("--sem-janela", action="store_true")
args = ap.parse_args()

CLASSE_VAZIA = "fundo"   # classe que significa "nenhum objeto"
DESCONHECIDO = "?"

model = tf.keras.models.load_model(args.modelo)
class_names = json.load(open("class_names.json"))

fonte = args.fonte
webcam = fonte.isdigit()
cap = cv2.VideoCapture(int(fonte) if webcam else fonte)
if not cap.isOpened():
    raise SystemExit(f"Nao consegui abrir: {fonte}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30

writer, saida, nome_base = None, None, "webcam"
if not webcam:
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nome_base = os.path.splitext(os.path.basename(fonte))[0]
    saida = f"saida_{nome_base}{args.sufixo}.mp4"
    writer = cv2.VideoWriter(saida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# [MELHORIA 1] buffer com as probabilidades dos ultimos N frames
historico = deque(maxlen=max(1, args.janela))
rotulos = []   # [MELHORIA 2] rotulo decidido em cada frame (para a linha do tempo)
n = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if webcam:
        frame = cv2.flip(frame, 1)

    img = cv2.cvtColor(letterbox(frame), cv2.COLOR_BGR2RGB).astype("float32")
    probs = model(img[None], training=False).numpy()[0]

    # [MELHORIA 1] suavizacao: media das probabilidades dos ultimos N frames
    historico.append(probs)
    probs_med = np.mean(historico, axis=0)
    i = int(np.argmax(probs_med))

    # [MELHORIA 1] limiar: se a maior probabilidade MEDIA for baixa, nao arrisca um nome
    rotulo = class_names[i] if probs_med[i] >= args.limiar else DESCONHECIDO
    rotulos.append(rotulo)

    # --- desenho ---
    if rotulo == DESCONHECIDO:
        texto, cor = f"? ({probs_med[i]:.0%})", (0, 0, 255)
    elif rotulo == CLASSE_VAZIA:
        texto, cor = f"nenhum Pokemon ({probs_med[i]:.0%})", (0, 200, 255)
    else:
        texto, cor = f"{rotulo} ({probs_med[i]:.0%})", (0, 200, 0)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, texto, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)
    for j, c in enumerate(class_names):
        y = 55 + j * 22
        cv2.rectangle(frame, (10, y), (10 + int(probs_med[j] * 150), y + 14), (0, 200, 255), -1)
        cv2.putText(frame, c, (170, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"t = {n / fps:5.1f}s | janela={args.janela} limiar={args.limiar:.0%}",
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if writer is not None:
        writer.write(frame)
    if webcam or not args.sem_janela:
        cv2.imshow("Reconhecimento de Pokemon", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    n += 1

cap.release()
if writer is not None:
    writer.release()
    print(f"Video anotado salvo em: {saida}")
cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# [MELHORIA 2] LINHA DO TEMPO
# ---------------------------------------------------------------------------
def montar_linha_do_tempo(rotulos, fps, min_trecho, lacuna):
    """Agrupa frames seguidos com o mesmo rotulo em trechos [rotulo, inicio_s, fim_s].
    Depois: (a) tapa buracos curtos entre dois trechos do mesmo Pokemon,
    (b) descarta '?', 'fundo' e trechos menores que min_trecho."""
    if not rotulos:
        return []
    trechos, ini = [], 0
    for k in range(1, len(rotulos) + 1):
        if k == len(rotulos) or rotulos[k] != rotulos[ini]:
            trechos.append([rotulos[ini], ini / fps, k / fps])   # tempo = indice / fps
            ini = k
    # (a) um "piscar" curto entre dois trechos iguais vira parte do trecho
    for k in range(1, len(trechos) - 1):
        ant, atual, prox = trechos[k - 1], trechos[k], trechos[k + 1]
        if ant[0] == prox[0] != atual[0] and (atual[2] - atual[1]) < lacuna:
            atual[0] = ant[0]
    fundidos = []
    for t in trechos:
        if fundidos and fundidos[-1][0] == t[0]:
            fundidos[-1][2] = t[2]
        else:
            fundidos.append(list(t))
    # (b) filtra
    return [t for t in fundidos
            if t[0] not in (DESCONHECIDO, CLASSE_VAZIA) and (t[2] - t[1]) >= min_trecho]


trechos = montar_linha_do_tempo(rotulos, fps, args.min_trecho, args.lacuna)
linhas = ["Linha do tempo (Pokemon identificados):"]
for rot, a, b in trechos:
    linhas.append(f"  {a:.1f}s a {b:.1f}s -> {rot}")
encontrados = sorted({t[0] for t in trechos})
linhas.append(f"Pokemon encontrados no video: {encontrados}")
n_duv = sum(r == DESCONHECIDO for r in rotulos)
linhas.append(f"(janela={args.janela}, limiar={args.limiar:.0%}; frames em '?': "
              f"{n_duv}/{len(rotulos)} = {n_duv / max(len(rotulos), 1):.0%})")

print("\n" + "\n".join(linhas))
arq = f"linha_do_tempo_{nome_base}{args.sufixo}.txt"
with open(arq, "w", encoding="utf-8") as f:
    f.write("\n".join(linhas) + "\n")
print(f"(linha do tempo tambem salva em {arq})")
