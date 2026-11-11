"""
Passo 2 - Treino com Transfer Learning (MobileNetV2) + FINE-TUNING.

Uso basico:   python train.py

Opcoes para os experimentos da Parte 5 (cada um gera arquivos com sufixo,
para nao sobrescrever o modelo principal):
  python train.py --split foto --tag split_foto        # Pergunta 1 (vazamento)
  python train.py --lr-ft 1e-3 --sem-early --tag ft_1e3 # Pergunta 2 (lr alta)
  python train.py --excluir pikachu:costas --tag sem_costas   # Pergunta 3

Fase 1: base congelada, so a "cabeca" nova aprende (lr 1e-3).
Fase 2 (FINE-TUNING, implementado pelo grupo): descongela as ultimas camadas
        da MobileNetV2 e continua com lr bem menor (1e-5).
A validacao e separada POR VIDEO (a menos que --split foto).
"""
import os
import json
import random
import argparse
from collections import defaultdict

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ argumentos
ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["video", "foto"], default="video",
                help="video = split correto (padrao); foto = sorteio por foto (Pergunta 1)")
ap.add_argument("--epocas1", type=int, default=8, help="epocas da fase 1 (cabeca)")
ap.add_argument("--epocas-ft", type=int, default=8, help="epocas da fase 2 (fine-tuning)")
ap.add_argument("--lr-ft", type=float, default=1e-5, help="taxa de aprendizado do fine-tuning")
ap.add_argument("--camadas-ft", type=int, default=30, help="quantas ultimas camadas da base descongelar")
ap.add_argument("--sem-ft", action="store_true", help="pula o fine-tuning")
ap.add_argument("--sem-early", action="store_true", help="desliga o EarlyStopping (para ver a loss subir)")
ap.add_argument("--excluir", default=None,
                help="classe:palavra -> ignora os videos da classe cujo nome contem a palavra (ex.: pikachu:costas)")
ap.add_argument("--tag", default="", help="sufixo dos arquivos de saida")
args = ap.parse_args()
SUF = f"_{args.tag}" if args.tag else ""

DATA_DIR = "data"
IMG_SIZE = 224
BATCH = 16
VAL_FRACTION = 0.2   # fracao de VIDEOS (nao de fotos) para validacao
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

class_names = sorted(d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, d)))
print("Classes:", class_names)

excl_cls, excl_palavra = None, None
if args.excluir:
    excl_cls, excl_palavra = args.excluir.split(":", 1)
    excl_palavra = excl_palavra.lower()
    if excl_cls not in class_names:
        raise SystemExit(f"--excluir: classe '{excl_cls}' nao existe em {class_names}")

# ------------------------------------------------------------------ split
# Frames do mesmo video sao quase iguais. Se um video tiver frames no treino e
# na validacao, a acuracia fica enganosamente alta (vazamento de dados).
# Por isso, no modo "video", cada video inteiro vai OU para treino OU validacao.
train_paths, train_y, val_paths, val_y = [], [], [], []
for idx, cls in enumerate(class_names):
    por_video = defaultdict(list)
    n_excluidas = 0
    for f in sorted(os.listdir(os.path.join(DATA_DIR, cls))):
        if not f.lower().endswith(".jpg"):
            continue
        video = f.split("__")[0]
        if cls == excl_cls and excl_palavra in video.lower():
            n_excluidas += 1          # [Pergunta 3] deixa de fora os frames "de costas"
            continue
        por_video[video].append(os.path.join(DATA_DIR, cls, f))
    if n_excluidas:
        print(f"  {cls}: {n_excluidas} frames excluidos ('{excl_palavra}')")

    videos = sorted(por_video)
    if len(videos) < 2:
        raise SystemExit(f"'{cls}' tem {len(videos)} video(s). Grave pelo menos 4 por classe.")

    if args.split == "video":
        random.Random(SEED).shuffle(videos)
        n_val = max(1, round(len(videos) * VAL_FRACTION))
        val_videos = set(videos[:n_val])
        for v in videos:
            destino = (val_paths, val_y) if v in val_videos else (train_paths, train_y)
            destino[0].extend(por_video[v]); destino[1].extend([idx] * len(por_video[v]))
        n_tr = sum(len(por_video[v]) for v in videos if v not in val_videos)
        n_va = sum(len(por_video[v]) for v in val_videos)
        print(f"  {cls}: {n_tr} treino / {n_va} validacao | videos de validacao: {sorted(val_videos)}")
    else:
        # [Pergunta 1] split ALEATORIO POR FOTO (errado de proposito: gera vazamento)
        todas = [p for v in videos for p in por_video[v]]
        random.Random(SEED).shuffle(todas)
        n_val = max(1, round(len(todas) * VAL_FRACTION))
        val_paths.extend(todas[:n_val]);  val_y.extend([idx] * n_val)
        train_paths.extend(todas[n_val:]); train_y.extend([idx] * (len(todas) - n_val))
        print(f"  {cls}: {len(todas) - n_val} treino / {n_val} validacao (split por FOTO)")

if args.split == "foto":
    print("\n*** ATENCAO: split por foto -> frames vizinhos do mesmo video em treino e "
          "validacao (vazamento). A acuracia de validacao vai parecer melhor do que e. ***\n")


def load(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img, label


def make_ds(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH)
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(train_paths, train_y, True)
val_ds = make_ds(val_paths, val_y, False)   # sem shuffle: a ordem bate com val_y

# Data augmentation (so atua no treino). O flip horizontal simula o objeto
# virado para o outro lado; a rotacao e o zoom simulam a camera se mexendo.
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.20),
    tf.keras.layers.RandomTranslation(0.10, 0.10),
    tf.keras.layers.RandomBrightness(0.25),
    tf.keras.layers.RandomContrast(0.25),
])

base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = augment(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)


def callbacks():
    if args.sem_early:
        return []
    return [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                             restore_best_weights=True)]


# ------------------------------------------------------------------ avaliacao
def avaliar(titulo, arquivo_png):
    """Imprime matriz de confusao + precisao/recall por classe e salva a matriz em PNG."""
    y_true = np.array(val_y)
    y_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)
    k = len(class_names)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    print(f"\n===== {titulo} =====")
    print("Matriz de confusao (linhas = real, colunas = previsto):")
    print(" " * 11 + " ".join(f"{c[:8]:>8}" for c in class_names))
    for i, c in enumerate(class_names):
        print(f"{c[:10]:>10} " + " ".join(f"{v:>8}" for v in cm[i]))
    print("\nClasse        precisao   recall")
    for i, c in enumerate(class_names):
        prec = cm[i, i] / max(cm[:, i].sum(), 1)
        rec = cm[i, i] / max(cm[i, :].sum(), 1)
        print(f"{c[:12]:<12} {prec:>9.2%} {rec:>8.2%}")
    acc = np.trace(cm) / cm.sum()
    print(f"\nAcuracia de validacao: {acc:.2%}")

    plt.figure(figsize=(5 + k * 0.4, 4 + k * 0.4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(k), class_names, rotation=45, ha="right"); plt.yticks(range(k), class_names)
    for i in range(k):
        for j in range(k):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xlabel("previsto"); plt.ylabel("real"); plt.title(titulo)
    plt.tight_layout(); plt.savefig(arquivo_png); plt.close()
    return acc


# ------------------------------------------------------------------ FASE 1: cabeca
print("\n########## FASE 1: base congelada (so a cabeca aprende) ##########")
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
hist1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epocas1, callbacks=callbacks())
model.save(f"modelo_fase1{SUF}.keras")
acc1 = avaliar("Matriz de confusao - Fase 1 (so a cabeca)", f"matriz_fase1{SUF}.png")

# ------------------------------------------------------------------ FASE 2: fine-tuning
hist2, acc2 = None, None
if not args.sem_ft:
    print(f"\n########## FASE 2: FINE-TUNING (ultimas {args.camadas_ft} camadas, lr={args.lr_ft}) ##########")
    # [MELHORIA 3 - fine-tuning]
    # 1) descongela a base inteira...
    base.trainable = True
    # 2) ...e congela de novo tudo, menos as ultimas N camadas
    for layer in base.layers[:-args.camadas_ft]:
        layer.trainable = False
    # 3) as BatchNormalization continuam congeladas (estabiliza o fine-tuning com poucos dados)
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    print("Camadas da base treinaveis:", sum(l.trainable for l in base.layers), "de", len(base.layers))
    # 4) OBRIGATORIO recompilar depois de mudar o que e treinavel, com lr bem menor
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr_ft),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epocas_ft, callbacks=callbacks())
    acc2 = avaliar("Matriz de confusao - Fase 2 (apos fine-tuning)", f"matriz_fase2{SUF}.png")

model.save(f"modelo{SUF}.keras")
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# ------------------------------------------------------------------ curvas (2 fases)
def junta(chave):
    return hist1.history[chave] + (hist2.history[chave] if hist2 else [])

n1 = len(hist1.history["loss"])
plt.figure(figsize=(11, 4))
for i, (m, t) in enumerate([("accuracy", "Acuracia"), ("loss", "Loss")]):
    plt.subplot(1, 2, i + 1)
    xs = range(1, len(junta(m)) + 1)
    plt.plot(xs, junta(m), label="treino")
    plt.plot(xs, junta("val_" + m), label="validacao")
    if hist2:
        plt.axvline(n1 + 0.5, color="gray", linestyle="--", label="inicio do fine-tuning")
    plt.title(t); plt.xlabel("epoca"); plt.legend()
plt.tight_layout(); plt.savefig(f"curvas{SUF}.png"); plt.close()

print("\n================ RESUMO ================")
print(f"Split: {args.split} | treino: {len(train_paths)} fotos | validacao: {len(val_paths)} fotos")
print(f"Acuracia de validacao - fase 1: {acc1:.2%}")
if acc2 is not None:
    print(f"Acuracia de validacao - fase 2 (fine-tuning, lr={args.lr_ft}): {acc2:.2%}")
print(f"Arquivos: modelo{SUF}.keras, modelo_fase1{SUF}.keras, curvas{SUF}.png, matriz_fase*{SUF}.png")

# ------------------------------------------------------------------ PDF do dataset
from PIL import Image, ImageDraw, ImageFont


def gerar_pdf_dataset(data_dir="data", saida="dataset_imagens.pdf", cols=10,
                      qualidade=70, max_por_classe=300):
    """Uma secao por classe. Se houver muitas fotos, pega uma amostra espacada
    (max_por_classe) para o PDF nao ficar gigante. A legenda mostra quantos
    videos geraram as fotos daquela classe."""
    pag_w, pag_h, margem, topo = 1240, 1754, 40, 110
    cell = (pag_w - 2 * margem) // cols
    rows = (pag_h - topo - margem) // cell
    por_pagina = cols * rows
    try:
        fonte = ImageFont.load_default(size=28)
    except TypeError:
        fonte = ImageFont.load_default()
    paginas = []
    classes = sorted(d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)))
    for classe in classes:
        pasta = os.path.join(data_dir, classe)
        fotos = sorted(f for f in os.listdir(pasta) if f.lower().endswith(".jpg"))
        n_videos = len({f.split("__")[0] for f in fotos})
        total = len(fotos)
        if total > max_por_classe:
            idx = np.linspace(0, total - 1, max_por_classe).astype(int)
            fotos = [fotos[i] for i in idx]
        n_pag = max(1, -(-len(fotos) // por_pagina))
        for p in range(n_pag):
            pagina = Image.new("RGB", (pag_w, pag_h), "white")
            draw = ImageDraw.Draw(pagina)
            draw.text((margem, 40),
                      f"Classe: {classe} | {total} imagens de {n_videos} videos | pagina {p + 1}/{n_pag}",
                      fill="black", font=fonte)
            for i, nome in enumerate(fotos[p * por_pagina:(p + 1) * por_pagina]):
                img = Image.open(os.path.join(pasta, nome)).convert("RGB")
                img.thumbnail((cell - 6, cell - 6))
                pagina.paste(img, (margem + (i % cols) * cell + 3, topo + (i // cols) * cell + 3))
            paginas.append(pagina)
    paginas[0].save(saida, save_all=True, append_images=paginas[1:],
                    quality=qualidade, resolution=150)
    print(f"PDF salvo: {saida} ({len(paginas)} paginas)")


if not args.tag:   # o PDF so e gerado no treino principal (nao nos experimentos)
    gerar_pdf_dataset(DATA_DIR)

print("\nPronto! Agora rode: python detectar.py meu_video_teste.mp4")
