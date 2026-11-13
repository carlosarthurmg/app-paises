"""Aplicativo de classificação de países por prioridade de ajuda.

Autor: Carlos Arthur Moraes Gonçalves
Inteligência Artificial · Unidade III · UNIFSA · Engenharia de Software · 2026.2
"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# 1. Configuração da página
st.set_page_config(page_title="Classificação de países", layout="centered")


# 2. Carregamento do modelo (executado uma única vez e mantido em cache)
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_paises.pkl")


pacote = carregar_modelo()
modelo = pacote["modelo"]      # pipeline: StandardScaler -> PCA -> KMeans
colunas = pacote["colunas"]    # ordem das colunas usada no treinamento
nomes = pacote["nomes"]        # nome de cada grupo
perfil = pacote.get("perfil")          # média de cada indicador por grupo
projecao = pacote.get("projecao")      # países do treino projetados em PC1 e PC2
info = pacote.get("info")              # metadados do treinamento

# 3. Descrição de cada campo: rótulo exibido e valor inicial (mediana da base)
CAMPOS = {
    "child_mort": ("Mortalidade infantil (mortes por 1.000 nascidos vivos)", 19.3),
    "exports": ("Exportações (% do PIB per capita)", 35.0),
    "health": ("Gasto com saúde (% do PIB per capita)", 6.3),
    "imports": ("Importações (% do PIB per capita)", 43.3),
    "income": ("Renda líquida por pessoa (US$)", 9960.0),
    "inflation": ("Inflação anual (%)", 5.4),
    "life_expec": ("Expectativa de vida (anos)", 73.1),
    "total_fer": ("Filhos por mulher", 2.4),
    "gdpp": ("PIB per capita (US$)", 4660.0),
}

# 4. Interface
st.title("Classificação de países por prioridade de ajuda")
st.write(
    "Aplicativo desenvolvido por **Carlos Arthur Moraes Gonçalves** para apoiar a "
    "HELP International na identificação de países com maior necessidade de ajuda. "
    "Informe os indicadores do país para obter o nível de prioridade estimado pelo modelo."
)

valores = {}
esquerda, direita = st.columns(2)
for i, coluna in enumerate(colunas):
    rotulo, inicial = CAMPOS[coluna]
    with esquerda if i % 2 == 0 else direita:
        valores[coluna] = st.number_input(rotulo, value=inicial, key=coluna)

# 5. Classificação
if st.button("Classificar", type="primary"):
    entrada = pd.DataFrame([valores])[colunas]   # uma linha, colunas na ordem do treinamento
    grupo = int(modelo.predict(entrada)[0])      # o pipeline padroniza, aplica o PCA e agrupa
    st.success(f"Grupo {grupo}: {nomes[grupo]}")

    # Valores informados (e, quando disponível, comparação com a média do grupo)
    tabela = pd.DataFrame(
        {"Valor informado": entrada.iloc[0]}
    )
    if perfil is not None:
        tabela["Média do grupo"] = perfil.loc[grupo, colunas]
        tabela["Diferença"] = tabela["Valor informado"] - tabela["Média do grupo"]
    tabela.index = [CAMPOS[c][0] for c in tabela.index]
    st.subheader("Valores informados" + (" e perfil típico do grupo" if perfil is not None else ""))
    st.dataframe(tabela.round(2))

    # Posição do país no plano PC1 x PC2 (padronização e PCA do próprio pipeline)
    if projecao is not None:
        z = modelo[:-1].transform(entrada)[0, :2]
        fig, ax = plt.subplots(figsize=(7, 5))
        for g, d in projecao.groupby("grupo"):
            ax.scatter(d["PC1"], d["PC2"], alpha=0.5, label=nomes[g])
        ax.scatter(z[0], z[1], marker="*", s=350, c="red", edgecolors="black",
                   label="País informado", zorder=5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        st.subheader("Posição no plano PC1 × PC2")
        st.pyplot(fig)

    # Aviso sobre limitações e uso responsável
    st.warning(
        "**Use este resultado apenas como apoio à decisão.**\n\n"
        "- Países próximos da fronteira entre dois grupos podem ter condições parecidas "
        "e receber classificações diferentes.\n"
        "- Os dados são de um conjunto público, podem estar desatualizados ou conter erros.\n"
        "- Os indicadores não captam conflitos, desastres naturais, desigualdade interna "
        "nem acesso a água e alimentos.\n"
        "- O agrupamento usa componentes principais, difíceis de explicar; justifique decisões "
        "com indicadores claros, como mortalidade infantil e renda.\n"
        "- Analise a situação atual do país antes de distribuir recursos."
    )

# Sobre o modelo (model card inicial)
st.divider()
st.header("Sobre o modelo")
if info is not None:
    st.markdown(
        f"- **Origem dos dados:** {info['origem']}\n"
        f"- **Data de treinamento:** {info['data_treino']}\n"
        f"- **Etapas do pipeline:** padronização (`StandardScaler`), PCA e K-Means\n"
        f"- **Componentes do PCA:** {info['n_componentes']} "
        f"(explicam {info['variancia_explicada']:.1%} da variância)\n"
        f"- **Número de grupos:** {info['n_grupos']}\n"
        f"- **Países usados no treino:** {info['n_paises']}"
    )
else:
    pca, kmeans = modelo[1], modelo[2]
    st.markdown(
        "- **Origem dos dados:** Kaggle – Unsupervised Learning on Country Data\n"
        f"- **Componentes do PCA:** {pca.n_components_}\n"
        f"- **Número de grupos:** {kmeans.n_clusters}"
    )