# =====================================================
# Trabalho Prático 01: Classificador KNN
# Disciplina: Inteligência Artificial
# Autor: Guilherme Noronha de Agostini e Gustavo Ribeiro de Figueiredo
# =====================================================
import time
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


# ==============================
# 1. Carregar dados direto do sklearn
# ==============================
def carregar_dados():
    iris = load_iris()  # Carrega o dataset já importado no sklearn

    # `iris.data` → matriz com as 4 features numéricas
    # `iris.target` → vetor com as classes (0,1,2)
    # `iris.target_names` → nomes reais das classes (setosa, versicolor, virginica)

    atributos = iris.data
    classes = iris.target
    nomes_classes = iris.target_names

    return atributos, classes, nomes_classes


# ==============================
# 2. Avaliação do modelo
# ==============================
def avaliar_classificador(knn, X_teste, y_teste):
    previsoes = knn.predict(X_teste)  # Faz as previsões

    return {
        "acuracia": accuracy_score(y_teste, previsoes) * 100,
        "revocacao": recall_score(y_teste, previsoes, average="weighted") * 100,
        "precisao": precision_score(y_teste, previsoes, average="weighted") * 100,
        "matriz_confusao": confusion_matrix(y_teste, previsoes)
    }


# ==============================
# 3. Execução principal
# ==============================
def main():
    inicio = time.time()

    # Carrega o dataset iris
    X, y, nomes_classes = carregar_dados()

    # Divide treino e teste (80/20)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n=== CLASSIFICADOR KNN (Scikit-Learn - Iris embutido) ===")
    print(f"Treino: {len(X_treino)} | Teste: {len(X_teste)}\n")

    k_opcoes = [1, 3, 5, 7]
    melhor_k, melhor_acc = None, 0

    for k in k_opcoes:
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X_treino, y_treino)

        resultados = avaliar_classificador(modelo, X_teste, y_teste)
        print(f"k = {k} | Acurácia: {resultados['acuracia']:.2f}% | "
              f"Revocação: {resultados['revocacao']:.2f}% | "
              f"Precisão: {resultados['precisao']:.2f}%")
        print("Matriz de Confusão:")
        print(resultados["matriz_confusao"], "\n")

        if resultados["acuracia"] > melhor_acc:
            melhor_acc, melhor_k = resultados["acuracia"], k

    print(f"\nMelhor valor de k: {melhor_k} com Acurácia {melhor_acc:.2f}%")
    print(f"Tempo total: {time.time() - inicio:.3f} segundos")


if __name__ == "__main__":
    main()
