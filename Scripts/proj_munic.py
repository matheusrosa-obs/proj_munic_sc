# ------------------------------------
# SCRIPT PARA PROJEÇÃO DO PIB DOS MUNICÍPIOS ATÉ 2030
# Modelo simples de distribuição das projeções do PIB das mesorregiões
# ------------------------------------
import polars as pl
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from warnings import filterwarnings

filterwarnings("ignore")

project_root = Path().resolve().parent
raw_path = project_root / 'Dados/Brutos'
interim_path = project_root / 'Dados/Limpos'
processed_path = project_root / 'Dados/Processados'
references_path = project_root / 'Referências'

# ------------------------------------
# CARREGAMENTO DA PROJEÇÃO DAS MESORREGIÕES ATÉ 2030
# ------------------------------------
df_pib_mesos_raw = pl.read_csv(interim_path / 'pib_mesos_projs.csv')

df_pib_mesos_raw.head()

# ------------------------------------
# CARREGAMENTO DOS DADOS BRUTOS DOS MUNICÍPIOS E DAS MESORREGIÕES
# ------------------------------------
df_munic_raw = pd.read_excel(raw_path / 'pib_munic.xlsx')
df_mesos_raw = pd.read_excel(raw_path / 'pib_mesos.xlsx')

df_munic_raw.head()
df_mesos_raw.head()

df_munic = df_munic_raw.set_index('munic').T

df_munic.head()

df_mesos = df_mesos_raw.set_index('ano')

df_munic.index = df_munic.index.astype(int)
df_mesos.index = df_mesos.index.astype(int)

df_munic = np.log(df_munic)
df_munic = df_munic.diff()

df_mesos = np.log(df_mesos)
df_mesos = df_mesos.diff()

df_mesos = df_mesos.rename(columns={'pib_oeste': 'Oeste Catarinense', 'pib_norte': 'Norte Catarinense',
                                    'pib_serrana': 'Serrana', 'pib_vale': 'Vale do Itajaí',
                                    'pib_granflor': 'Grande Florianópolis', 'pib_sul': 'Sul Catarinense',
                                    'pib_sc': 'Santa Catarina'})

df_mesos = df_mesos[df_mesos.index >= 2003]

df_munic = df_munic.reset_index().melt(id_vars='index', var_name='Munic', value_name='pib_munic')
df_munic = df_munic.rename(columns={'index': 'Ano'})
df_munic = df_munic.sort_values(by=['Munic', 'Ano'])
df_munic = df_munic[df_munic['Ano'] >= 2003]

df_mesos_map = pd.read_excel(references_path / 'munic_mesos.xlsx')

df_munic['Mesorregião'] = df_munic['Munic'].map(df_mesos_map.set_index('Munic')['Mesos'])

df_munic = df_munic[['Ano', 'Munic', 'Mesorregião', 'pib_munic']]

#df_munic = df_munic.dropna()

df_munic['pib_mesoregiao'] = df_munic.apply(
    lambda row: df_mesos.loc[row['Ano'], row['Mesorregião']] if row['Mesorregião'] in df_mesos.columns else np.nan,
    axis=1)

# Executa regressão para cada município e armazena os resultados em um dicionário
regression_results = {}

for munic, group in df_munic.dropna(subset=['pib_munic', 'pib_mesoregiao']).groupby('Munic'):
    model = sm.glm('pib_munic ~ pib_mesoregiao', data=group).fit(cov_type='HC0')
    regression_results[munic] = model

# Exemplo: acessar o resumo da regressão de um município específico
print(regression_results['Florianópolis'].summary())


proj_mesos = pd.read_excel(interim_path / 'pib_mesos_projs.xlsx')
proj_mesos = proj_mesos.set_index('ano')
proj_mesos = np.log(proj_mesos)
proj_mesos = proj_mesos.diff()
proj_mesos = proj_mesos[(proj_mesos.index >= 2003) & (proj_mesos.index <= 2030)]

proj_mesos = proj_mesos.rename(columns={'pib_oeste': 'Oeste Catarinense', 'pib_norte': 'Norte Catarinense',
                                    'pib_serrana': 'Serrana', 'pib_vale': 'Vale do Itajaí',
                                    'pib_granflor': 'Grande Florianópolis', 'pib_sul': 'Sul Catarinense',
                                    'pib_sc': 'Santa Catarina'})

# Prever valores de pib_munic para cada município usando os valores projetados das mesos em proj_mesos

# Lista para armazenar previsões
munic_forecasts = []

# Para cada município com regressão estimada
for munic, model in regression_results.items():
    # Descobrir a mesorregião do município
    meso = df_mesos_map.loc[df_mesos_map['Munic'] == munic, 'Mesos'].values
    if len(meso) == 0 or meso[0] not in proj_mesos.columns:
        continue  # pula se não encontrar mesorregião ou coluna correspondente
    meso = meso[0]
    
    # Seleciona os anos e valores projetados da mesorregião
    mesos_proj = proj_mesos[meso]
    anos_proj = mesos_proj.index

    # Monta DataFrame para previsão
    df_pred = pd.DataFrame({
        'Ano': anos_proj,
        'pib_mesoregiao': mesos_proj.values
    })

    # Faz a previsão usando o modelo estimado
    pred = model.predict(df_pred)
    df_pred['Munic'] = munic
    df_pred['pib_munic_pred'] = pred

    munic_forecasts.append(df_pred[['Ano', 'Munic', 'pib_munic_pred']])

# Junta todas as previsões em um único DataFrame
df_munic_forecast = pd.concat(munic_forecasts, ignore_index=True)

df_munic_forecast = df_munic_forecast.pivot(index='Munic', columns='Ano', values='pib_munic_pred')

df_munic_forecast.to_excel(processed_path / 'pib_munic_forecast.xlsx')

