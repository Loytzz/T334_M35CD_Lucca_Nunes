import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_white
from statsmodels.graphics.gofplots import qqplot
import warnings
warnings.filterwarnings('ignore')

# =============================================
# 1. PRÉ-PROCESSAMENTO DOS DADOS 
# =============================================
print("1. Pré-processamento dos dados...")


df = pd.read_csv('dataset_19.csv')


df['latencia_ms'].fillna(df['latencia_ms'].median(), inplace=True)
df['armazenamento_tb'].fillna(df['armazenamento_tb'].median(), inplace=True)
df['tipo_hd'].fillna(df['tipo_hd'].mode()[0], inplace=True)
df['tipo_processador'].fillna(df['tipo_processador'].mode()[0], inplace=True)


df_encoded = pd.get_dummies(
    data=df,
    columns=['sistema_operacional', 'tipo_hd', 'tipo_processador'],
    drop_first=True,  
    dtype=int  
)


for col in df_encoded.columns:
    df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')


df_encoded.dropna(inplace=True)

# =============================================
# 2. MODELO DE REGRESSÃO LINEAR
# =============================================
print("\n2. Construção do modelo de regressão...")


X = df_encoded.drop('tempo_resposta', axis=1)
y = df_encoded['tempo_resposta']


X = sm.add_constant(X)


print("\nVerificação final de dados:")
print(f"- NaN em X: {X.isna().sum().sum()}")
print(f"- NaN em y: {y.isna().sum()}")

# Ajustar modelo
modelo = sm.OLS(y, X).fit()

# =============================================
# 3. RESULTADOS DO MODELO
# =============================================
print("\n3. Resultados do modelo completo:")
print(modelo.summary())

# =============================================
# 4. DIAGNÓSTICO DE MULTICOLINEARIDADE (VIF)
# =============================================
print("\n4. Diagnóstico de multicolinearidade (VIF):")


vif_data = pd.DataFrame()
vif_data["Variável"] = X.columns.drop('const')
vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                  for i in range(1, X.shape[1])]  

print(vif_data.sort_values("VIF", ascending=False))

# =============================================
# 5. DIAGNÓSTICO DE HETEROCEDASTICIDADE
# =============================================
print("\n5. Diagnóstico de heterocedasticidade:")

labels = ['Estatística', 'Valor-p', 'F-Statistic', 'F p-value']
white_test = het_white(modelo.resid, modelo.model.exog)
print(dict(zip(labels, white_test)))


plt.figure(figsize=(10, 6))
plt.scatter(modelo.fittedvalues, modelo.resid, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Resíduos vs Valores Ajustados')
plt.xlabel('Valores Ajustados')
plt.ylabel('Resíduos')
plt.savefig('residuos_vs_ajustados.png', bbox_inches='tight')
plt.close()

# =============================================
# 6. ANÁLISE DE RESÍDUOS
# =============================================
print("\n6. Análise de resíduos:")

# QQ-Plot
plt.figure(figsize=(10, 6))
qqplot(modelo.resid, line='s')
plt.title('Q-Q Plot dos Resíduos')
plt.savefig('qqplot_residuos.png', bbox_inches='tight')
plt.close()

# Histograma dos resíduos
plt.figure(figsize=(10, 6))
sns.histplot(modelo.resid, kde=True)
plt.title('Distribuição dos Resíduos')
plt.savefig('distribuicao_residuos.png', bbox_inches='tight')
plt.close()

# =============================================
# 7. MODELO REDUZIDO 
# =============================================
print("\n7. Modelo reduzido (exemplo removendo variáveis com VIF > 5):")


variaveis_selecionadas = vif_data[vif_data['VIF'] < 5]['Variável']
X_reduzido = X[['const'] + list(variaveis_selecionadas)]


modelo_reduzido = sm.OLS(y, X_reduzido).fit()

print("\nResultados do modelo reduzido:")
print(modelo_reduzido.summary())

print(f"\nComparação:")
print(f"- Modelo completo (R² ajustado): {modelo.rsquared_adj:.4f}")
print(f"- Modelo reduzido (R² ajustado): {modelo_reduzido.rsquared_adj:.4f}")

print("\nAnálise concluída! Gráficos salvos no diretório atual.")
