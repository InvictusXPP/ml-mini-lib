# ml-mini-lib

Mini biblioteka Machine Learning napisana w Pythonie, implementująca
prostą sieć neuronową (feedforward) z **ręcznie zaimplementowanym backpropagation**  
oraz **opcjonalnym przyspieszeniem GPU (CUDA / CuPy)**.

Projekt został stworzony w celach edukacyjnych – do nauki:
- działania sieci neuronowych „od zera”
- różnic między CPU i GPU
- implementacji podstawowych algorytmów optymalizacji

---

## ✨ Funkcjonalności

- ✅ Sieć neuronowa typu **2 → hidden → 1**
- ✅ Ręczny **forward pass** i **backpropagation**
- ✅ Aktywacje: `tanh`, `sigmoid`
- ✅ Funkcja straty: **MSE**
- ✅ Optymalizatory:
  - SGD
  - SGD z momentum
  - Adam
- ✅ Backend:
  - **CPU (NumPy)** – zawsze dostępny
  - **GPU (CuPy + CUDA)** – opcjonalny
- ✅ Wizualizacja:
  - wykres strat (loss)
  - granica decyzyjna XOR
- ✅ Test numeryczny gradientów
- ✅ Zapis i odczyt wag modelu

---

## 🧠 Przykład problemu – XOR

Biblioteka demonstruje rozwiązanie klasycznego problemu XOR:

| x1 | x2 | y |
|----|----|---|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

XOR **nie jest liniowo separowalny**, dlatego wymaga warstwy ukrytej.

---

## 📁 Struktura projektu

ml-mini-lib/

│

├── mllib/ # biblioteka ML

│ ├── backend.py

│ ├── tensor_ops.py

│ ├── layers.py

│ ├── model.py

│ ├── optimizers.py

│ ├── training.py

│ ├── utils.py

│ └── viz.py

│

├── apps/

│ ├── xor_demo/ # demo XOR

│ ├── benchmark/ # CPU vs GPU

│ └── playground/ # eksperymenty

│

├── setup.py

├── pyproject.toml

├── requirements.txt

└── README.md


yaml
Skopiuj kod

---

## ⚙️ Wymagania

### Podstawowe (CPU)
- Python ≥ 3.9
- NumPy
- Matplotlib

### Opcjonalne (GPU)
- NVIDIA GPU
- CUDA
- CuPy (`cupy-cuda12x`)

---

## 🚀 Instalacja (zalecane: virtualenv)

### 1️⃣ Utworzenie i aktywacja venv

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
2️⃣ Instalacja zależności
bash
Skopiuj kod
pip install -r requirements.txt
3️⃣ Instalacja biblioteki (tryb developerski)
bash
Skopiuj kod
pip install -e .
▶️ Uruchomienie demo XOR
bash
Skopiuj kod
python apps/xor_demo/main.py
Po treningu model powinien poprawnie klasyfikować XOR:

css
Skopiuj kod
Input  ->  Prediction
[0, 0] -> 0
[0, 1] -> 1
[1, 0] -> 1
[1, 1] -> 0
🧪 CPU vs GPU
Backend wybierany jest jawnie:

python
Skopiuj kod
net = SimpleFFN(2, 8, 1, backend="cpu")  # NumPy
net = SimpleFFN(2, 8, 1, backend="gpu")  # CuPy (jeśli dostępne)
Jeśli GPU/CUDA nie jest dostępne, biblioteka automatycznie działa na CPU.

📊 Wizualizacje
Wykres spadku funkcji straty (loss vs epoch)

Granica decyzyjna wyuczona przez sieć neuronową

🧪 Test gradientów
Biblioteka zawiera numeryczne sprawdzanie gradientów w celu
weryfikacji poprawności backpropagation.

📌 Cel projektu
Projekt ma charakter edukacyjny i służy do:

nauki ML „od podstaw”

zrozumienia matematyki sieci neuronowych

porównania CPU vs GPU

przygotowania pod dalsze rozszerzenia (CNN, Softmax, Cross-Entropy)

🧩 Możliwe rozszerzenia

Softmax + CrossEntropy

Batch Normalization (pełna wersja)

Convolutional layers

Autograd

Eksport do ONNX
