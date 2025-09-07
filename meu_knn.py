# =====================================================
# Trabalho Prático 01: Classificador KNN
# Disciplina: Inteligência Artificial
# Autor: Guilherme Noronha de Agostini e Gustavo Ribeiro de Figueiredo
# =====================================================
import math
import random
import time
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, recall_score, precision_score

# ==============================
# 1. Carregar dados do sklearn
# ==============================
""" Carrega o dataset Iris do Scikit-Learn e o prepara. """
def carregar_dados():
    iris = load_iris()
    atributos = iris.data
    classes = iris.target
    # Combina atributos e classe numa lista de listas
    dados = [list(atributos[i]) + [classes[i]] for i in range(len(classes))]
    return dados, iris.target_names

# ==============================
# 2. Distância Euclidiana
# ==============================
"""  Calcula a distância euclidiana entre dois registros (excluindo a classe) """
def distancia_euclidiana(registro1, registro2):
    # Considera apenas as 4 colunas numéricas (ignora a última coluna = classe)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(registro1[:-1], registro2[:-1])))

# ==============================
# 3. Predição via votação majoritária
# ==============================
""" Prevê a classe de um exemplo com base nos k vizinhos mais próximos."""
def prever_classe(exemplo, base_treino, k):
    # Calcula distância de 'exemplo' para todos os dados de treino
    distancias = [(distancia_euclidiana(exemplo, ref), ref[-1]) for ref in base_treino]

    # Ordena pela distância e pega os k vizinhos mais próximos
    vizinhos = sorted(distancias, key=lambda x: x[0])[:k]

    # Contagem de votos usando Counter para otimizar
    votos = Counter(classe for _, classe in vizinhos)

    # Retorna a classe mais votada
    return max(votos, key=votos.get)

# ==============================
# 4. Avaliação do modelo
# ==============================
"""Cria a matriz de confusão manualmente."""
def criar_matriz_confusao(classes_reais, classes_previstas):
    classes = sorted(list(set(classes_reais) | set(classes_previstas)))
    num_classes = len(classes)
    matriz = [[0] * num_classes for _ in range(num_classes)]
    
    mapeamento_classe_indice = {cls: i for i, cls in enumerate(classes)}
    
    for indice_real, indice_previsto in zip(classes_reais, classes_previstas):
        indice_real_na_matriz = mapeamento_classe_indice[indice_real]
        indice_previsto_na_matriz = mapeamento_classe_indice[indice_previsto]
        matriz[indice_real_na_matriz][indice_previsto_na_matriz] += 1
        
    return matriz

"""Calcula as métricas de desempenho e a matriz de confusão."""
def avaliar_modelo(base_teste, base_treino, k):

    classes_reais = [instancia[-1] for instancia in base_teste]
    classes_previstas = [prever_classe(instancia, base_treino, k) for instancia in base_teste]

    return {
        "acuracia": accuracy_score(classes_reais, classes_previstas) * 100,
        "revocacao": recall_score(classes_reais, classes_previstas, average="weighted") * 100,
        "precisao": precision_score(classes_reais, classes_previstas, average="weighted") * 100,
        "matriz_confusao": criar_matriz_confusao(classes_reais, classes_previstas)
    }

# ==============================
# 5. Divisão treino/teste
# ==============================
"""Embaralha e divide os dados em treino e teste."""
def dividir_dados(dados, proporcao_teste=0.2):
    random.seed(42) # Garante a mesma divisão para reprodutibilidade
    random.shuffle(dados)  # embaralha os registros
    limite = int(len(dados) * proporcao_teste)
    return dados[limite:], dados[:limite]  # retorna treino e teste

# ==============================
# 6. Execução principal
# ==============================
"""Função principal que orquestra a execução do classificador."""
def main():
    inicio = time.time()

    # Carrega o dataset
    base_completa, nomes_classes = carregar_dados()

    # Separa treino e teste
    treino, teste = dividir_dados(base_completa, proporcao_teste=0.2)

    print("\n=== CLASSIFICADOR KNN (Implementado Manualmente) ===")
    print(f"Total de exemplos: {len(base_completa)}")
    print(f"Conjunto de Treino: {len(treino)} | Conjunto de Teste: {len(teste)}\n")

    opcoes_k = [1, 3, 5, 7]
    melhor_k, melhor_acuracia = None, 0

    for k in opcoes_k:
        resultados = avaliar_modelo(teste, treino, k)
        print(f"k = {k} | Acurácia: {resultados['acuracia']:.2f}% | "
              f"Revocação: {resultados['revocacao']:.2f}% | "
              f"Precisão: {resultados['precisao']:.2f}%")
        
        print("Matriz de Confusão:")
        for linha in resultados["matriz_confusao"]:
            print(linha)
        print("\n")

        if resultados["acuracia"] > melhor_acuracia:
            melhor_acuracia, melhor_k = resultados["acuracia"], k

    print(f"\nMelhor valor de k: {melhor_k} com Acurácia {melhor_acuracia:.2f}%")
    print(f"Tempo total de execução: {time.time() - inicio:.3f} segundos")

if __name__ == "__main__":
    main()