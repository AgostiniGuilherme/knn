# Classificador KNN - Análise e Implementação

Este repositório contém o código-fonte de um trabalho prático sobre o algoritmo de classificação K-Nearest Neighbors (KNN). O projeto foi desenvolvido com duas abordagens principais: uma **implementação manual (do zero)** e uma **implementação utilizando a biblioteca Scikit-Learn**. O objetivo é comparar o desempenho, as métricas de avaliação e o tempo de execução entre as duas abordagens.

---

## 🧐 O Que É KNN?

O **K-Nearest Neighbors (KNN)**, ou K-Vizinhos Mais Próximos, é um dos algoritmos de aprendizado de máquina mais simples e intuitivos. Ele é usado para problemas de classificação e regressão.

Em um problema de classificação, o KNN funciona da seguinte maneira:

1.  **Armazenamento de Dados**: O algoritmo apenas armazena o conjunto de dados de treinamento.
2.  **Cálculo da Distância**: Para classificar um novo ponto de dados, o algoritmo calcula a distância (geralmente Euclidiana) desse novo ponto para todos os pontos no conjunto de treinamento.
3.  **Identificação dos Vizinhos**: Ele seleciona os `k` vizinhos mais próximos (os `k` pontos com as menores distâncias).
4.  **Votação**: O novo ponto é classificado com base na classe mais comum entre esses `k` vizinhos. A classe que "recebeu mais votos" é a classe final atribuída ao novo ponto.

---

## 📁 Estrutura do Projeto

* `meu_knn.py`: Contém a implementação manual do algoritmo KNN. As funções de cálculo de distância, predição e avaliação do modelo foram desenvolvidas do zero.
* `meu_knn_sklearn.py`: Utiliza a biblioteca `sklearn` para a implementação do KNN, servindo como referência para comparação de desempenho.
* `requirements.txt`: Lista todas as bibliotecas Python necessárias para executar o projeto.
* `relatorio.pdf`: Documento com a análise e os resultados detalhados do projeto.

---

## 📊 Análise e Resultados

O projeto utiliza a famosa base de dados **Iris**, que contém informações sobre as características de três espécies de flores (setosa, versicolor, e virginica).

Os resultados da análise comparativa demonstram que:

* **Acurácia**: A versão do Scikit-Learn obteve um desempenho ligeiramente superior, alcançando 100% de acurácia para alguns valores de `k`, enquanto a versão manual obteve um máximo de 96.67%.
* **Tempo de Execução**: A biblioteca otimizada do Scikit-Learn foi significativamente mais rápida (quase duas vezes mais rápida que a manual), reforçando sua eficiência para uso em larga escala.

A implementação manual foi crucial para a compreensão didática do algoritmo. No entanto, o uso de bibliotecas otimizadas é a melhor prática para projetos reais, dada a sua superioridade em desempenho e confiabilidade.

---

## 🚀 Como Executar

Para rodar os scripts e replicar o ambiente do projeto, siga os passos abaixo:

1.  **Clone este repositório:**
    ```bash
    git clone https://github.com/AgostiniGuilherme/knn.git
    cd knn
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    # Para sistemas Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # Para sistemas macOS e Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Instale as dependências do projeto:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute os scripts:**
    ```bash
    python meu_knn.py
    python meu_knn_sklearn.py
    ```
