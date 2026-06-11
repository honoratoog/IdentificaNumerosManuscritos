"""
Classificação de Dígitos Manuscritos com MLP
Disciplina: Inteligência Artificial - Engenharia de Software

Referências:
  - https://www.kaggle.com/code/jonathankristanto/mnist-classification-using-multilayer-perceptron
  - https://www.kaggle.com/c/digit-recognizer/data

Dependências:
    pip install scikit-learn matplotlib numpy pillow
    (Tkinter já vem incluído no Python padrão)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import threading
import os
import pickle

# =============================================================================
# 1. CARREGAMENTO E TREINAMENTO DO MODELO
# =============================================================================

MODEL_FILE = "mnist_mlp_model.pkl"

def carregar_ou_treinar_modelo(status_callback=None):
    """
    Carrega modelo salvo em disco ou treina um novo com o dataset MNIST.
    O sklearn baixa o MNIST automaticamente via fetch_openml.
    """
    # Se já existe modelo salvo, carrega direto
    if os.path.exists(MODEL_FILE):
        if status_callback:
            status_callback("Carregando modelo salvo...")
        with open(MODEL_FILE, "rb") as f:
            dados = pickle.load(f)
        if status_callback:
            status_callback(f"Modelo carregado! Acurácia: {dados['acuracia']:.2%}")
        return dados["modelo"], dados["acuracia"]

    # Caso contrário, treina do zero
    from sklearn.datasets import fetch_openml
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    if status_callback:
        status_callback("Baixando dataset MNIST (pode demorar na primeira vez)...")

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data, mnist.target.astype(int)

    # Normaliza para [0, 1]
    X = X / 255.0

    if status_callback:
        status_callback("Dividindo dados em treino/teste (80/20)...")

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if status_callback:
        status_callback("Treinando MLP... (aguarde, isso leva alguns minutos)")

    # Arquitetura MLP conforme especificação:
    # entrada (784) → camada oculta 1 (1024) → camada oculta 2 (64) → saída (10)
    modelo = MLPClassifier(
        hidden_layer_sizes=(1024, 64),
        activation="relu",
        solver="adam",
        max_iter=50,
        learning_rate_init=0.001,
        random_state=42,
        verbose=False,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=8,
    )

    modelo.fit(X_treino, y_treino)

    acuracia = modelo.score(X_teste, y_teste)

    if status_callback:
        status_callback(f"Treinamento concluído! Acurácia: {acuracia:.2%}")

    # Salva modelo para uso futuro
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"modelo": modelo, "acuracia": acuracia}, f)

    return modelo, acuracia


# =============================================================================
# 2. PRÉ-PROCESSAMENTO DA IMAGEM DO CANVAS
# =============================================================================

def preprocessar_canvas(imagem_pil):
    """
    Converte a imagem desenhada no canvas para o formato esperado pelo modelo.

    Etapas (replicando o pipeline do MNIST original):
      1. Escala de cinza
      2. Crop automático no dígito (remove bordas vazias)
      3. Adiciona margem e centraliza pelo centro de massa
      4. Redimensiona para 20x20 e cola em canvas 28x28 centralizado
      5. Suavização leve + normalização [0, 1]
    """
    from PIL import ImageOps, ImageFilter
    import numpy as np

    img = imagem_pil.convert("L")

    # --- Dilata levemente o traço para ficar mais parecido com escrita humana ---
    img = img.filter(ImageFilter.MaxFilter(size=3))

    arr = np.array(img)

    # --- Crop: encontra bounding box do dígito ---
    linhas = np.any(arr > 30, axis=1)
    colunas = np.any(arr > 30, axis=0)

    if not linhas.any():
        # Canvas vazio
        img_28 = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
        return np.zeros((1, 784), dtype=np.float32), img_28

    r_min, r_max = np.where(linhas)[0][[0, -1]]
    c_min, c_max = np.where(colunas)[0][[0, -1]]

    # Crop no dígito
    cropped = arr[r_min:r_max+1, c_min:c_max+1]

    # --- Redimensiona para caber em 20x20 mantendo proporção ---
    h, w = cropped.shape
    escala = 20.0 / max(h, w)
    novo_h = max(1, int(round(h * escala)))
    novo_w = max(1, int(round(w * escala)))

    img_crop = Image.fromarray(cropped)
    img_resized = img_crop.resize((novo_w, novo_h), Image.LANCZOS)

    # --- Cola centralizado em canvas 28x28 ---
    canvas_28 = Image.new("L", (28, 28), 0)
    offset_x = (28 - novo_w) // 2
    offset_y = (28 - novo_h) // 2
    canvas_28.paste(img_resized, (offset_x, offset_y))

    # --- Centraliza pelo centro de massa (como o MNIST faz) ---
    arr28 = np.array(canvas_28, dtype=np.float32)
    total = arr28.sum()
    if total > 0:
        cx = int((arr28 * np.arange(28)).sum(axis=1).sum() / total)
        cy = int((arr28 * np.arange(28).reshape(28, 1)).sum(axis=1).sum() / total)
        shift_x = 14 - cy
        shift_y = 14 - cx
        from PIL import Image as PILImage
        canvas_28 = PILImage.fromarray(arr28.astype(np.uint8))
        canvas_28 = canvas_28.transform(
            (28, 28), Image.AFFINE, (1, 0, -shift_x, 0, 1, -shift_y), fillcolor=0
        )

    # --- Suavização final ---
    canvas_28 = canvas_28.filter(ImageFilter.GaussianBlur(radius=0.5))

    arr_final = np.array(canvas_28, dtype=np.float32) / 255.0

    return arr_final.reshape(1, -1), canvas_28


# =============================================================================
# 3. INTERFACE GRÁFICA (TKINTER)
# =============================================================================

class AppMNIST:
    CANVAS_SIZE = 280      # Canvas de desenho (280x280 px)
    PREVIEW_SIZE = 112     # Preview 28x28 ampliado para 112x112

    def __init__(self, root):
        self.root = root
        self.root.title("Classificador de Dígitos Manuscritos — MLP")
        self.root.resizable(False, False)

        self.modelo = None
        self.acuracia_modelo = 0.0

        # Imagem PIL para armazenar o que foi desenhado
        self.imagem_pil = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 0)
        self.draw_pil = ImageDraw.Draw(self.imagem_pil)

        self._construir_ui()
        self._carregar_modelo_async()

    # ------------------------------------------------------------------
    # Construção da interface
    # ------------------------------------------------------------------

    def _construir_ui(self):
        pad = dict(padx=10, pady=6)

        # --- Título ---
        tk.Label(
            self.root,
            text="Classificador de Dígitos Manuscritos",
            font=("Helvetica", 15, "bold"),
        ).pack(**pad)

        tk.Label(
            self.root,
            text="Multilayer Perceptron (MLP) treinado no MNIST",
            font=("Helvetica", 10),
            fg="#555555",
        ).pack()

        # --- Frame principal ---
        frame_principal = tk.Frame(self.root)
        frame_principal.pack(padx=12, pady=8)

        # Coluna esquerda: canvas de desenho
        frame_esq = tk.Frame(frame_principal)
        frame_esq.grid(row=0, column=0, padx=(0, 16))

        tk.Label(frame_esq, text="Desenhe um dígito (0–9):", font=("Helvetica", 10)).pack(anchor="w")

        self.canvas = tk.Canvas(
            frame_esq,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            bg="black",
            cursor="crosshair",
        )
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>", self._ao_desenhar)
        self.canvas.bind("<ButtonRelease-1>", self._ao_soltar)

        # Botões
        frame_botoes = tk.Frame(frame_esq)
        frame_botoes.pack(fill="x", pady=(6, 0))

        self.btn_classificar = tk.Button(
            frame_botoes,
            text="Classificar",
            font=("Helvetica", 11, "bold"),
            bg="#1a73e8",
            fg="black",
            activebackground="#1558b0",
            activeforeground="white",
            width=12,
            command=self._classificar,
        )
        self.btn_classificar.pack(side="left", padx=(0, 8))

        tk.Button(
            frame_botoes,
            text="Limpar",
            font=("Helvetica", 11),
            width=10,
            command=self._limpar,
        ).pack(side="left")

        # Coluna direita: resultado + preview
        frame_dir = tk.Frame(frame_principal)
        frame_dir.grid(row=0, column=1, sticky="n")

        # Resultado
        tk.Label(frame_dir, text="Resultado:", font=("Helvetica", 10)).pack(anchor="w")

        self.label_digito = tk.Label(
            frame_dir,
            text="—",
            font=("Helvetica", 72, "bold"),
            fg="#1a73e8",
            width=3,
        )
        self.label_digito.pack()

        self.label_confianca = tk.Label(
            frame_dir,
            text="Confiança: —",
            font=("Helvetica", 10),
            fg="#555555",
        )
        self.label_confianca.pack()

        # Preview 28x28
        tk.Label(frame_dir, text="Entrada da rede (28×28):", font=("Helvetica", 9), fg="#777777").pack(
            anchor="w", pady=(12, 2)
        )

        self.canvas_preview = tk.Canvas(
            frame_dir,
            width=self.PREVIEW_SIZE,
            height=self.PREVIEW_SIZE,
            bg="#111111",
            highlightthickness=1,
            highlightbackground="#cccccc",
        )
        self.canvas_preview.pack()

        # Barras de probabilidade
        tk.Label(frame_dir, text="Probabilidades:", font=("Helvetica", 9), fg="#777777").pack(
            anchor="w", pady=(12, 2)
        )

        self.frame_barras = tk.Frame(frame_dir)
        self.frame_barras.pack(fill="x")
        self._criar_barras_prob()

        # --- Status bar ---
        self.var_status = tk.StringVar(value="Inicializando modelo...")
        tk.Label(
            self.root,
            textvariable=self.var_status,
            font=("Helvetica", 9),
            fg="#888888",
            anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 8))

    def _criar_barras_prob(self):
        self.barras = []
        self.labels_prob = []
        for i in range(10):
            linha = tk.Frame(self.frame_barras)
            linha.pack(fill="x", pady=1)

            tk.Label(linha, text=str(i), font=("Helvetica", 9), width=2).pack(side="left")

            barra_bg = tk.Frame(linha, bg="#e0e0e0", height=10, width=100)
            barra_bg.pack(side="left", padx=2)
            barra_bg.pack_propagate(False)

            barra_fill = tk.Frame(barra_bg, bg="#1a73e8", height=10, width=0)
            barra_fill.place(x=0, y=0, height=10)

            lbl = tk.Label(linha, text="0.0%", font=("Helvetica", 8), fg="#555555", width=5)
            lbl.pack(side="left")

            self.barras.append(barra_fill)
            self.labels_prob.append(lbl)

    # ------------------------------------------------------------------
    # Carregamento assíncrono do modelo
    # ------------------------------------------------------------------

    def _carregar_modelo_async(self):
        def tarefa():
            try:
                modelo, acuracia = carregar_ou_treinar_modelo(
                    status_callback=lambda msg: self.var_status.set(msg)
                )
                self.modelo = modelo
                self.acuracia_modelo = acuracia
                self.var_status.set(
                    f"Modelo pronto — Acurácia no teste: {acuracia:.2%}   |   "
                    f"Arquitetura: entrada(784) → 1024 → 64 → saída(10)"
                )
            except Exception as e:
                self.var_status.set(f"Erro ao carregar modelo: {e}")

        t = threading.Thread(target=tarefa, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Eventos do canvas
    # ------------------------------------------------------------------

    def _ao_desenhar(self, evento):
        raio = 10
        x, y = evento.x, evento.y

        # Desenha no canvas Tkinter (visual)
        self.canvas.create_oval(
            x - raio, y - raio, x + raio, y + raio,
            fill="white", outline="white"
        )

        # Desenha na imagem PIL (para processar depois)
        self.draw_pil.ellipse(
            [x - raio, y - raio, x + raio, y + raio],
            fill=255
        )

    def _ao_soltar(self, _evento):
        pass

    # ------------------------------------------------------------------
    # Ações dos botões
    # ------------------------------------------------------------------

    def _limpar(self):
        self.canvas.delete("all")
        self.imagem_pil = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 0)
        self.draw_pil = ImageDraw.Draw(self.imagem_pil)

        self.label_digito.config(text="—")
        self.label_confianca.config(text="Confiança: —")
        self.canvas_preview.delete("all")

        for i in range(10):
            self.barras[i].place_configure(width=0)
            self.labels_prob[i].config(text="0.0%")

    def _classificar(self):
        if self.modelo is None:
            messagebox.showwarning("Aguarde", "O modelo ainda está sendo carregado. Tente em instantes.")
            return

        # Pré-processa
        entrada, img_28 = preprocessar_canvas(self.imagem_pil)

        # Predição
        probabilities = self.modelo.predict_proba(entrada)[0]
        digito_predito = int(np.argmax(probabilities))
        confianca = probabilities[digito_predito]

        # Atualiza resultado
        self.label_digito.config(text=str(digito_predito))
        self.label_confianca.config(text=f"Confiança: {confianca:.1%}")

        # Atualiza preview 28x28
        self._atualizar_preview(img_28)

        # Atualiza barras de probabilidade
        for i in range(10):
            prob = probabilities[i]
            largura = int(prob * 100)
            cor = "#1a73e8" if i == digito_predito else "#90caf9"
            self.barras[i].place_configure(width=largura)
            self.barras[i].config(bg=cor)
            self.labels_prob[i].config(text=f"{prob:.1%}")

    def _atualizar_preview(self, img_28):
        """Renderiza o 28x28 ampliado no canvas de preview."""
        from tkinter import PhotoImage

        arr = np.array(img_28)
        escala = self.PREVIEW_SIZE // 28  # = 4

        self.canvas_preview.delete("all")

        for row in range(28):
            for col in range(28):
                valor = int(arr[row, col])
                if valor > 10:
                    hex_cor = f"#{valor:02x}{valor:02x}{valor:02x}"
                    x0 = col * escala
                    y0 = row * escala
                    self.canvas_preview.create_rectangle(
                        x0, y0, x0 + escala, y0 + escala,
                        fill=hex_cor, outline=""
                    )


# =============================================================================
# 4. PONTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = AppMNIST(root)
    root.mainloop()