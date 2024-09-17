import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Importar dados
df = pd.read_csv('medical_examination.csv')

# 2. Adicionar a coluna 'overweight'
df['BMI'] = df['weight'] / (df['height'] / 100) ** 2
df['overweight'] = (df['BMI'] > 25).astype(int)

# 3. Normalizar as colunas 'cholesterol' e 'gluc'
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4. Função para desenhar o gráfico categórico
def draw_cat_plot():
    # 5. Criar DataFrame para o gráfico categórico usando `pd.melt`
    df_cat = pd.melt(df, id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Agrupar e reformatar os dados para dividir por 'cardio' e mostrar a contagem de cada característica
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()

    # Renomear a coluna 'size' para 'total'
    df_cat = df_cat.rename(columns={'size': 'total'})
    
    # 7. Desenhar o gráfico categórico usando `sns.catplot()`
    catplot = sns.catplot(x='variable', y='total', hue='value', col='cardio', kind='bar', data=df_cat)

    # 8. Salvar a figura
    catplot.fig.savefig('catplot.png')
    return catplot.fig

# 9. Função para desenhar o Heatmap
def draw_heat_map():
    # 10. Limpar os dados
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 11. Calcular a matriz de correlação (somente colunas numéricas e relevantes)
    corr = df_heat[['id', 'age', 'sex', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'overweight']].corr()

    # 12. Gerar uma máscara para o triângulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 13. Configurar a figura do matplotlib
    fig, ax = plt.subplots(figsize=(12, 8))

    # 14. Desenhar o Heatmap com o método `sns.heatmap()`
    sns.heatmap(corr, annot=True, mask=mask, fmt='.1f', center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.5}, ax=ax)

    # 15. Salvar a figura do Heatmap
    fig.savefig('heatmap.png')
    return fig
