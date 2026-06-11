# 🔢 Classificador de Dígitos Manuscritos — MLP

Projeto desenvolvido para a disciplina de **Inteligência Artificial** do curso de Engenharia de Software.

Implementação de uma Rede Neural do tipo **Multilayer Perceptron (MLP)** treinada no dataset MNIST para classificação de dígitos manuscritos (0–9), com interface interativa onde o usuário desenha o dígito com o mouse e a IA classifica em tempo real.

---

## 🧠 Arquitetura da Rede Neural

```
Entrada:          784 neurônios   (imagem 28×28 pixels)
Camada oculta 1: 1024 neurônios   (ativação ReLU)
Camada oculta 2:   64 neurônios   (ativação ReLU)
Saída:             10 neurônios   (dígitos 0 a 9)
```

- **Otimizador:** Adam  
- **Early stopping:** sim (evita overfitting)  
- **Acurácia no conjunto de teste:** ~97–98%

---

## 📁 Estrutura do Projeto

```
├── mnist_mlp.py          # Código principal: treinamento + interface gráfica
├── mnist_mlp_model.pkl   # Modelo salvo após o primeiro treinamento (gerado automaticamente)
└── README.md
```

---

## ⚙️ Instalação e Execução

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install scikit-learn numpy pillow pandas
```

### 4. Execute

```bash
python3 mnist_mlp.py
```

> Na **primeira execução**, o programa baixa o dataset MNIST automaticamente e treina o modelo (leva alguns minutos). Nas execuções seguintes, o modelo já fica salvo em `mnist_mlp_model.pkl` e carrega instantaneamente.

---

## 🖥️ Como usar a interface

1. **Desenhe** um dígito de 0 a 9 na área preta com o mouse
2. Clique em **Classificar**
3. Veja o resultado, a confiança da rede e as probabilidades para cada dígito
4. Clique em **Limpar** para desenhar novamente

---

## 🛠️ Tecnologias utilizadas

| Biblioteca | Uso |
|---|---|
| `scikit-learn` | MLPClassifier, download do MNIST, divisão treino/teste |
| `NumPy` | Manipulação de arrays, normalização, centro de massa |
| `Pillow` | Pré-processamento da imagem desenhada |
| `Tkinter` | Interface gráfica (canvas, botões, barras de probabilidade) |
| `pickle` | Serialização do modelo treinado |
| `pandas` | Dependência do fetch_openml para carregar o MNIST |

---

## 🔄 Pré-processamento da imagem

Para minimizar a diferença entre o desenho feito com o mouse e as imagens do MNIST, o sistema aplica o mesmo pipeline de pré-processamento usado no dataset original:

1. Dilatação do traço (aproxima da espessura da escrita humana)
2. Crop automático removendo bordas vazias
3. Redimensionamento para 20×20 mantendo a proporção
4. Centralização em canvas 28×28
5. Centralização pelo **centro de massa** do dígito
6. Normalização dos pixels para `[0.0, 1.0]`

---

## 📚 Referências

- [MNIST Classification using Multilayer Perceptron — Kaggle](https://www.kaggle.com/code/jonathankristanto/mnist-classification-using-multilayer-perceptron)
- [Digit Recognizer — Kaggle Competition](https://www.kaggle.com/c/digit-recognizer/data)
- [scikit-learn MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

---

## 👨‍💻 Autor

Feito por **Lucas Honorato dos Santos** — Engenharia de Software  
Disciplina: Inteligência Artificial
