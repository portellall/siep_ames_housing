# -----------------------------------------------------------------------------
# Web App de Análise de Dados Imobiliários com Streamlit
# -----------------------------------------------------------------------------
#
# Autor: Seu Assistente de IA
# Data: 16/10/2025
#
# DESCRIÇÃO:
# Este aplicativo permite a análise interativa do Ames Housing Dataset.
# Ele foi projetado para fins acadêmicos, focando em visualização de dados,
# ANOVA e Regressão Linear Múltipla para entender os fatores que
# influenciam o preço de venda dos imóveis.
#
# INSTRUÇÕES DE EXECUÇÃO:
# 1. Certifique-se de que você tem o Python instalado.
# 2. Salve este arquivo como `app.py`.
# 3. Crie um arquivo chamado `requirements.txt` com o conteúdo abaixo.
# 4. Instale as bibliotecas executando no terminal:
#    pip install -r requirements.txt
# 5. Para iniciar o aplicativo, execute o seguinte comando no terminal:
#    streamlit run app.py
# 6. Use a barra lateral para fazer o upload do arquivo `AmesHousing.csv`.
#
# -----------------------------------------------------------------------------

# Importação das bibliotecas necessárias
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

# Ignorar avisos para uma apresentação mais limpa
warnings.filterwarnings('ignore')

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Análise de Preços de Imóveis | Ames Housing",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNÇÃO DE CARREGAMENTO DE DADOS (COM CACHE) ---
@st.cache_data
def load_data(file):
    """
    Carrega os dados de um arquivo (local ou uploaded) e faz uma limpeza básica.
    """
    df = pd.read_csv(file)
    # Renomear colunas para facilitar o manuseio
    df.columns = df.columns.str.replace(' ', '_').str.replace('/', '_')
    # Corrigir tipos de dados que podem ser problemáticos
    if 'PID' in df.columns:
        df['PID'] = df['PID'].astype(str)
    if 'Garage_Yr_Blt' in df.columns:
        # Preenche NAs com o ano de construção da casa
        df['Garage_Yr_Blt'].fillna(df['Year_Built'], inplace=True)
    return df

# --- TÍTULO E INTRODUÇÃO ---
st.title("🏠 Analisador de Preços de Imóveis de Ames")
st.markdown("""
Esta aplicação interativa foi desenvolvida para analisar o *Ames Housing Dataset*.
Utilize as abas abaixo para explorar os dados, realizar análises estatísticas e construir
modelos de regressão para entender os fatores que influenciam o preço dos imóveis.
""")

# --- UPLOADER DE ARQUIVO NA SIDEBAR ---
st.sidebar.header("1. Carregar Dados")
uploaded_file = st.sidebar.file_uploader(
    "Faça o upload do seu arquivo CSV aqui",
    type=["csv"]
)

df = None # Inicializa o dataframe como nulo

if uploaded_file is not None:
    # Se o usuário carregar um arquivo, use-o
    df = load_data(uploaded_file)
    st.sidebar.success("Arquivo carregado com sucesso!")
else:
    # Senão, tente carregar o arquivo local
    try:
        df = load_data('AmesHousing.csv')
        st.sidebar.info("Dataset `AmesHousing.csv` carregado da pasta local.")
    except FileNotFoundError:
        st.warning("Para começar, por favor, carregue o arquivo `AmesHousing.csv` usando o botão acima.")

# --- O RESTANTE DA APLICAÇÃO SÓ RODA SE O DATAFRAME FOR CARREGADO ---
if df is not None:
    # --- IDENTIFICAÇÃO DAS COLUNAS ---
    # Colunas que são numéricas, mas devem ser tratadas como categóricas
    potential_cats_as_numeric = ['MS_SubClass', 'Overall_Qual', 'Overall_Cond', 'Mo_Sold']
    
    # Seleciona colunas numéricas (que não sejam as categóricas acima)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in potential_cats_as_numeric + ['Order', 'PID', 'SalePrice']]
    
    # Seleciona colunas de objeto (texto) e as numéricas que são tratadas como categóricas
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist() + potential_cats_as_numeric
    
    # --- INTERFACE COM ABAS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Visão Geral do Dataset",
        "📊 Visualização Exploratória",
        "📈 Análise de Variância (ANOVA)",
        "📉 Regressão Linear Múltipla"
    ])

    # --- ABA 1: VISÃO GERAL DO DATASET ---
    with tab1:
        st.header("Explorando o Conjunto de Dados")
        st.markdown("Aqui você encontra uma visão geral dos dados, incluindo as primeiras linhas, dimensões e estatísticas descritivas.")

        st.subheader("Amostra dos Dados")
        st.dataframe(df.head())

        st.subheader("Dimensões do Dataset")
        st.info(f"O conjunto de dados possui **{df.shape[0]} linhas** e **{df.shape[1]} colunas**.")

        st.subheader("Estatísticas Descritivas para Variáveis Numéricas")
        st.dataframe(df[numeric_cols + ['SalePrice']].describe())

        st.subheader("Resumo de Variáveis Categóricas")
        st.dataframe(df[categorical_cols].describe())

    # --- ABA 2: VISUALIZAÇÃO EXPLORATÓRIA ---
    with tab2:
        st.header("Análise Gráfica Interativa")
        st.sidebar.header("2. Opções de Visualização")

        plot_type = st.sidebar.radio(
            "Escolha o tipo de gráfico:",
            ("Histograma", "Boxplot", "Gráfico de Dispersão", "Mapa de Calor de Correlação")
        )

        if plot_type == "Histograma":
            st.subheader("Distribuição de uma Variável Numérica")
            var = st.selectbox("Selecione a variável numérica:", numeric_cols + ['SalePrice'])
            fig = px.histogram(df, x=var, title=f'Histograma de {var}', nbins=50, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Boxplot":
            st.subheader("Distribuição de uma Variável Numérica por Categoria")
            num_var = st.selectbox("Selecione a variável numérica:", numeric_cols + ['SalePrice'])
            cat_var = st.selectbox("Selecione a variável categórica:", categorical_cols)
            fig = px.box(df, x=cat_var, y=num_var, title=f'Boxplot de {num_var} por {cat_var}', template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Gráfico de Dispersão":
            st.subheader("Relação entre Duas Variáveis Numéricas")
            x_var = st.selectbox("Selecione a variável para o eixo X:", numeric_cols + ['SalePrice'])
            y_var = st.selectbox("Selecione a variável para o eixo Y:", numeric_cols + ['SalePrice'], index=1)
            color_var = st.selectbox("Selecione uma variável para a cor (opcional):", [None] + categorical_cols)
            fig = px.scatter(df, x=x_var, y=y_var, color=color_var, title=f'Dispersão: {x_var} vs {y_var}', template="plotly_white", trendline="ols",
                             trendline_color_override="red")
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Mapa de Calor de Correlação":
            st.subheader("Correlação entre Variáveis Numéricas")
            selected_vars = st.multiselect(
                "Selecione pelo menos duas variáveis:",
                numeric_cols + ['SalePrice'],
                default=['SalePrice', 'Gr_Liv_Area', 'Total_Bsmt_SF', 'Year_Built']
            )
            if len(selected_vars) > 1:
                corr_matrix = df[selected_vars].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                title="Mapa de Calor de Correlação", color_continuous_scale='RdBu_r')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Por favor, selecione pelo menos duas variáveis.")

    # --- ABA 3: ANÁLISE DE VARIÂNCIA (ANOVA) ---
    with tab3:
        st.header("Análise de Variância (ANOVA) de Uma Via")
        st.markdown("""
        A ANOVA é usada para testar se há diferenças estatisticamente significativas entre as médias de dois ou mais grupos.
        Aqui, vamos testar se o preço médio de venda (`SalePrice`) difere entre as categorias de uma variável que você escolher.
        """)

        # Seleção da variável categórica
        st.sidebar.header("3. Opções da ANOVA")
        cat_var_anova = st.sidebar.selectbox(
            "Selecione a variável categórica (independente):",
            [col for col in categorical_cols if df[col].nunique() > 1 and df[col].nunique() < 30], # Evita variáveis com muitas categorias
            index=categorical_cols.index('Neighborhood')
        )
        alpha = st.sidebar.slider("Nível de Significância (α):", 0.01, 0.10, 0.05, 0.01)

        st.subheader(f"Analisando 'SalePrice' por '{cat_var_anova}'")

        # Preparar dados
        data_anova = df[['SalePrice', cat_var_anova]].dropna()
        groups = [data_anova['SalePrice'][data_anova[cat_var_anova] == g] for g in data_anova[cat_var_anova].unique()]

        # --- VERIFICAÇÃO DO NÚMERO DE GRUPOS ---
        if len(groups) < 2:
            st.warning(f"""
            A variável selecionada ('{cat_var_anova}') possui menos de dois grupos com dados válidos 
            após a remoção de valores ausentes. A análise ANOVA e outros testes comparativos 
            requerem pelo menos dois grupos. 
            
            **Por favor, selecione outra variável categórica na barra lateral.**
            """)
        else:
            # Visualização
            fig_box = px.box(data_anova, x=cat_var_anova, y='SalePrice', title=f'Distribuição de SalePrice por {cat_var_anova}')
            st.plotly_chart(fig_box, use_container_width=True)

            st.markdown("---")
            st.subheader("Verificação dos Pressupostos")

            col1, col2 = st.columns(2)

            # 1. Normalidade (Shapiro-Wilk)
            with col1:
                st.markdown("##### 1. Normalidade dos Resíduos por Grupo")
                normality_passed = True
                for i, g in enumerate(data_anova[cat_var_anova].unique()):
                    # Adiciona um check para garantir que o grupo tenha dados suficientes para o teste
                    if len(groups[i]) >= 3:
                        shapiro_test = stats.shapiro(groups[i])
                        if shapiro_test.pvalue < alpha:
                            normality_passed = False
                            st.error(f"Grupo '{g}': p-valor = {shapiro_test.pvalue:.4f}. **Não Normal**. (Rejeita H0)")
                        else:
                            st.success(f"Grupo '{g}': p-valor = {shapiro_test.pvalue:.4f}. **Normal**. (Não Rejeita H0)")
                    else:
                        st.warning(f"Grupo '{g}': Não possui dados suficientes (mínimo 3) para o teste de Shapiro-Wilk.")
                        normality_passed = False # Considera falha se não pode ser testado
                
                if normality_passed:
                    st.info("O pressuposto de normalidade foi atendido para todos os grupos testáveis.")
                else:
                    st.warning("O pressuposto de normalidade **falhou** para pelo menos um grupo ou não pôde ser testado.")

            # 2. Homocedasticidade (Levene)
            with col2:
                st.markdown("##### 2. Homogeneidade das Variâncias (Levene)")
                levene_test = stats.levene(*groups)
                homoscedasticity_passed = levene_test.pvalue >= alpha
                if homoscedasticity_passed:
                    st.success(f"p-valor = {levene_test.pvalue:.4f}. As variâncias são homogêneas. (Não Rejeita H0)")
                else:
                    st.error(f"p-valor = {levene_test.pvalue:.4f}. As variâncias **não** são homogêneas. (Rejeita H0)")

            st.markdown("---")
            st.subheader("Resultados do Teste Estatístico")

            # Escolha do teste
            if normality_passed and homoscedasticity_passed:
                st.info("Os pressupostos foram atendidos. Realizando teste ANOVA.")
                f_val, p_val = stats.f_oneway(*groups)
                st.metric("Estatística F", f"{f_val:.4f}")
                st.metric("P-valor", f"{p_val:.4f}")

                # Interpretação
                if p_val < alpha:
                    st.success(f"**Conclusão:** Como o p-valor ({p_val:.4f}) é menor que o nível de significância ({alpha}), "
                               f"rejeitamos a hipótese nula. Há evidências de que o preço médio de venda difere "
                               f"significativamente entre as diferentes categorias de '{cat_var_anova}'.")
                else:
                    st.warning(f"**Conclusão:** Como o p-valor ({p_val:.4f}) é maior que o nível de significância ({alpha}), "
                               f"não rejeitamos a hipótese nula. Não há evidências suficientes para afirmar que o preço médio "
                               f"de venda difere entre as categorias de '{cat_var_anova}'.")
            else:
                st.warning("Como um ou mais pressupostos da ANOVA não foram atendidos, o teste não paramétrico de Kruskal-Wallis é mais apropriado.")
                h_val, p_val = stats.kruskal(*groups)
                st.metric("Estatística H (Kruskal-Wallis)", f"{h_val:.4f}")
                st.metric("P-valor", f"{p_val:.4f}")

                # Interpretação
                if p_val < alpha:
                    st.success(f"**Conclusão:** O teste de Kruskal-Wallis indica que há uma diferença estatisticamente "
                               f"significativa na distribuição dos preços de venda entre as categorias de '{cat_var_anova}' "
                               f"(p-valor = {p_val:.4f} < {alpha}).")
                else:
                    st.warning(f"**Conclusão:** O teste de Kruskal-Wallis não encontrou uma diferença estatisticamente "
                               f"significativa na distribuição dos preços de venda entre as categorias de '{cat_var_anova}' "
                               f"(p-valor = {p_val:.4f} >= {alpha}).")

    # --- ABA 4: REGRESSÃO LINEAR MÚLTIPLA ---
    with tab4:
        st.header("Modelo de Regressão Linear Múltipla")
        st.markdown("""
        Construa um modelo para prever `SalePrice` com base em múltiplas variáveis.
        Selecione as variáveis independentes (explicativas) nos painéis abaixo. O modelo
        cuidará automaticamente da criação de variáveis *dummy* para as categóricas.
        """)
        st.sidebar.header("4. Opções da Regressão")

        # Seleção de variáveis
        selected_numeric = st.sidebar.multiselect(
            "Selecione as variáveis numéricas:", numeric_cols, default=['Gr_Liv_Area', 'Total_Bsmt_SF']
        )
        selected_categorical = st.sidebar.multiselect(
            "Selecione as variáveis categóricas:", categorical_cols, default=['Overall_Qual', 'Neighborhood']
        )

        if not selected_numeric or not selected_categorical:
            st.warning("Por favor, selecione pelo menos uma variável numérica e uma categórica para construir o modelo.")
        else:
            # Construir a fórmula
            predictors = selected_numeric + [f'C({col})' for col in selected_categorical]
            formula = f'SalePrice ~ {" + ".join(predictors)}'
            st.info(f"**Fórmula do Modelo:** `{formula}`")

            # Preparar dados
            data_reg = df[['SalePrice'] + selected_numeric + selected_categorical].dropna()

            # Ajustar o modelo
            try:
                model = ols(formula, data=data_reg).fit()
                
                st.subheader("Sumário do Modelo")
                st.code(model.summary())

                st.subheader("Métricas de Desempenho do Modelo")
                predictions = model.predict(data_reg)
                rmse = np.sqrt(mean_squared_error(data_reg['SalePrice'], predictions))
                mae = mean_absolute_error(data_reg['SalePrice'], predictions)
                
                col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                col_metric1.metric("R²", f"{model.rsquared:.3f}")
                col_metric2.metric("R² Ajustado", f"{model.rsquared_adj:.3f}")
                col_metric3.metric("RMSE", f"${rmse:,.2f}")
                col_metric4.metric("MAE", f"${mae:,.2f}")

                st.subheader("Interpretação Automática dos Coeficientes")
                st.markdown("A seguir, a interpretação para os coeficientes estatisticamente significativos (p-valor < 0.05):")
                significant_params = model.pvalues[model.pvalues < 0.05].index
                for param in significant_params:
                    if param != 'Intercept':
                        coef_val = model.params[param]
                        st.success(f"**{param}:** Mantendo todas as outras variáveis constantes, um aumento de uma unidade em `{param}` está associado a um **{'aumento' if coef_val > 0 else 'decréscimo'}** de **${abs(coef_val):,.2f}** no preço de venda.")
                
                st.subheader("Análise de Resíduos")
                residuals = model.resid
                fitted = model.fittedvalues
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown("##### Normalidade dos Resíduos (Gráfico Q-Q)")
                    fig_qq = sm.qqplot(residuals, line='s', fit=True)
                    st.pyplot(fig_qq)
                    st.caption("Os pontos devem seguir a linha vermelha para indicar normalidade.")
                    
                with res_col2:
                    st.markdown("##### Homocedasticidade (Resíduos vs. Valores Previstos)")
                    fig_res = px.scatter(x=fitted, y=residuals, labels={'x': 'Valores Previstos', 'y': 'Resíduos'},
                                         title="Resíduos vs. Valores Previstos", trendline="lowess", trendline_color_override="red")
                    fig_res.add_hline(y=0, line_dash="dash", line_color="black")
                    st.plotly_chart(fig_res, use_container_width=True)
                    st.caption("Para homocedasticidade, os pontos devem se espalhar aleatoriamente em torno da linha horizontal, sem formar padrões.")

            except Exception as e:
                st.error(f"Ocorreu um erro ao ajustar o modelo: {e}")
                st.warning("Isso pode ser causado por multicolinearidade perfeita ou categorias com poucos dados. Tente selecionar um conjunto diferente de variáveis.")