import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from warnings import filterwarnings

filterwarnings("ignore")

project_root = Path().resolve().parent
raw_path = project_root / 'data/raw'
interim_path = project_root / 'data/interim'
processed_path = project_root / 'data/processed'
references_path = project_root / 'references'





################# PROJEÇÃO PIB DE SC #################
df_pib_sc_raw = pd.read_excel(raw_path / 'pib_sc.xlsx')
df_pib_sc_raw = df_pib_sc_raw.set_index('ano')
df_pib_sc_raw = df_pib_sc_raw.map(lambda x: float(str(x).replace(',', '.')))

df_pib_sc_raw['selic_over_1'] = df_pib_sc_raw['selic_over'].shift(1)
df_pib_sc_raw['cambio_uss_1'] = df_pib_sc_raw['cambio_uss'].shift(1)
df_pib_sc_raw = df_pib_sc_raw.dropna()

reg_elast = sm.glm('np.log(pib_sc) ~ np.log(pib_br) + np.log(pib_usa) + np.log(pib_chn) + selic_over_1 +np.log(cambio_uss_1)',
                   data=df_pib_sc_raw).fit(cov_type='HC0')

print(reg_elast.summary())

'''
CENÁRIO-BASE:
BRASIL:
    Em 2025, 2,5% de crescimento do PIB no Brasil, a partir do consumo das famílias e da boa perspectiva para a safra agrícola.
    Em 2026, 2% de crescimento. Deve pesar negativamente o impacto em defasagem do aperto monetário, mas fiscal em ano eleitoral deve ter impacto positivo no crescimento.
    Em 2027, 1,5% de crescimento. A desaceleração deve aparecer devido a necessidade de correção fiscal no primeiro ano do novo governo.
    A partir de 2028, crescimento estabilizado em 2,0%, seguindo as projeções do Focus.
EUA:
    Em 2025, crescimento de 1,2%. Apesar da demanda interna aquecida, o aperto monetário deve continuar segurando o crescimento. As incertezas de comércio contribuem para a desaceleração.
    Em 2026, crescimento de 1,5%. A provável inflação decorrente das tarifas de importação deve incitar a continuidade do ciclo de alta dos juros, condicionando o crescimento lento.
    A partir de 2027, crescimento estabilizado em 2,0%.
CHINA:
    Em todos os anos, projeções de acordo com o Euro Monitor.
SELIC:
    Em 2025, expectativa de 13,75%a.a. na selic meta. A selic over deve seguir a trajetória de aperto monetário até o final do primeiro semestre, com possível relaxamento no segundo.
    Em 2026, ano pré-eleitoral e com impactos consolidados na inflação (espera-se) deve haver espaço para o início de um ciclo de queda da selic meta.
    A partir de 2027, projeções do Boletim Focus.
CAMBIO USS:
    Todas as projeções são do Boletim Focus.
'''

taxas_base = pd.DataFrame({
    'ano': range(2025, 2031),
    'pib_br': [0.025, 0.02, 0.015, 0.02, 0.02, 0.02],
    'pib_usa': [0.012, 0.015, 0.02, 0.02, 0.02, 0.02],
    'pib_chn': [0.045, 0.042, 0.037, 0.036, 0.034, 0.034],
})

taxas_pess = pd.DataFrame({
    'ano': range(2025, 2031),
    'pib_br': [0.02, 0.01, 0.015, 0.015, 0.02, 0.02],
    'pib_usa': [0.01, 0.015, 0.01, 0.01, 0.02, 0.02],
    'pib_chn': [0.03, 0.02, 0.02, 0.02, 0.03, 0.03],
})

taxas_otim = pd.DataFrame({
    'ano': range(2025, 2031),
    'pib_br': [0.035, 0.03, 0.025, 0.025, 0.02, 0.02],
    'pib_usa': [0.02, 0.025, 0.03, 0.03, 0.02, 0.02],
    'pib_chn': [0.05, 0.04, 0.04, 0.04, 0.03, 0.03],
})

taxas_base = taxas_base.set_index('ano')
taxas_pess = taxas_pess.set_index('ano')
taxas_otim = taxas_otim.set_index('ano')

ultimo_ano = df_pib_sc_raw.index[-1]
valores_atuais = df_pib_sc_raw.loc[ultimo_ano, ['pib_br', 'pib_usa', 'pib_chn']]

### AQUI SE SELECIONA O CENÁRIO DE PROJEÇÃO DAS VARIÁVEIS EXPLICATIVAS
taxas = taxas_base.copy()

projs_sc = pd.DataFrame(index=taxas.index, columns=['pib_br', 'pib_usa', 'pib_chn'])

for ano in taxas.index:
    crescimento = taxas.loc[ano]
    valores_atuais = valores_atuais*(1 + crescimento)
    projs_sc.loc[ano] = valores_atuais

projs_sc = projs_sc.astype(float)

selic_base = [10.83, 14, 12.5, 10.5, 10.0, 10.0]
selic_otim = [10.83, 12.5, 11.5, 10.5, 8, 8]
selic_pess = [10.83, 14.5, 13, 12, 10.0, 10.0] 

### AQUI SE SELECIONA O CENÁRIO DE PROJEÇÃO TAXA SELIC
projs_sc.loc[2025:, 'selic_over_1'] = selic_base
projs_sc.loc[2025:, 'cambio_uss_1'] = [5.91, 5.85, 5.9, 5.8, 5.82, 5.8]

projs_sc['pib_sc'] = np.exp(reg_elast.predict(projs_sc))

df_concat_sc = pd.concat([df_pib_sc_raw, projs_sc])
df_concat_sc = df_concat_sc[df_concat_sc.index >= 2010]
df_concat_sc_taxas = df_concat_sc.pct_change(fill_method=None)

df_concat_sc.to_excel(interim_path / 'pib_sc_projs.xlsx', index=True)
df_concat_sc_taxas.to_excel(interim_path / 'pib_sc_projs_taxas.xlsx', index=True)







################# PROJEÇÃO PIB DAS MESORREGIÕES #################
df_pib_mesos_raw = pd.read_excel(raw_path / 'pib_mesos.xlsx')
df_pib_mesos_raw = df_pib_mesos_raw.set_index('ano')

## EXTRAINDO OS LOGARITMOS E AS DIFERENÇAS
df_pib_mesos_raw_logs = np.log(df_pib_mesos_raw)
df_pib_mesos_raw_logs_difs = df_pib_mesos_raw_logs.diff().dropna()

reg_elast_granflor = sm.glm('pib_granflor ~ pib_sc', data=df_pib_mesos_raw_logs_difs).fit(cov_type='HC0')
reg_elast_oeste = sm.glm('pib_oeste ~ pib_sc', data=df_pib_mesos_raw_logs_difs).fit(cov_type='HC0')
reg_elast_norte = sm.glm('pib_norte ~ pib_sc', data=df_pib_mesos_raw_logs_difs).fit(cov_type='HC0')
reg_elast_serrana = sm.glm('pib_serrana ~ pib_sc', data=df_pib_mesos_raw_logs_difs).fit(cov_type='HC0')
reg_elast_vale = sm.glm('pib_vale ~ pib_sc', data=df_pib_mesos_raw_logs_difs).fit(cov_type='HC0')
reg_elast_sul = sm.glm('pib_sul ~ pib_sc', data=df_pib_mesos_raw_logs_difs).fit(cov_type='HC0')

print(reg_elast_serrana.summary())

regs = {
    'pib_granflor': reg_elast_granflor,
    'pib_oeste': reg_elast_oeste,
    'pib_norte': reg_elast_norte,
    'pib_serrana': reg_elast_serrana,
    'pib_vale': reg_elast_vale,
    'pib_sul': reg_elast_sul,
}

elasticidades_df = pd.DataFrame({reg: model.params for reg, model in regs.items()})

projs_mesos_raw = pd.read_excel(project_root / 'data/interim/pib_sc_projs.xlsx')
projs_mesos_raw = projs_mesos_raw.set_index('ano')

projs_mesos_logs = np.log(projs_mesos_raw)

projs_mesos_logs_difs = projs_mesos_logs.diff()

projs_mesos_logs_difs

projs_mesos = pd.DataFrame(index=range(2022, 2031), columns=['pib_sc'])

for ano in projs_mesos.index:
    projs_mesos.loc[ano, 'pib_sc'] = projs_mesos_logs_difs.loc[ano, 'pib_sc']

projs_mesos = projs_mesos.astype(float)

projs_mesos['pib_granflor'] = np.exp(reg_elast_granflor.predict(projs_mesos))-1
projs_mesos['pib_oeste'] = np.exp(reg_elast_oeste.predict(projs_mesos))-1
projs_mesos['pib_norte'] = np.exp(reg_elast_norte.predict(projs_mesos))-1
projs_mesos['pib_serrana'] = np.exp(reg_elast_serrana.predict(projs_mesos))-1
projs_mesos['pib_vale'] = np.exp(reg_elast_vale.predict(projs_mesos))-1
projs_mesos['pib_sul'] = np.exp(reg_elast_sul.predict(projs_mesos))-1

### RECONSTITUIÇÃO DA SÉRIE EM NÍVEL
pib_mesos_proj = df_pib_mesos_raw.copy()

for col in ['pib_granflor', 'pib_oeste', 'pib_norte', 'pib_serrana', 'pib_vale', 'pib_sul']:
    ultimo_valor = df_pib_mesos_raw.loc[df_pib_mesos_raw.index.max(), col]
    for ano in projs_mesos.index:
        crescimento = projs_mesos.loc[ano, col]
        novo_valor = ultimo_valor*(1 + crescimento)
        pib_mesos_proj.loc[ano, col] = novo_valor
        ultimo_valor = novo_valor

pib_mesos_proj = pib_mesos_proj.sort_index()

pib_mesos_proj_taxas = pib_mesos_proj.pct_change(fill_method=None)

pib_mesos_proj.to_excel(interim_path / 'pib_mesos_projs.xlsx', index=True)
pib_mesos_proj_taxas.to_excel(interim_path / 'pib_mesos_projs_taxas.xlsx', index=True)









################# PROJEÇÃO PIB DOS MUNICÍPIOS #################
df_munic_raw = pd.read_excel(raw_path / 'pib_munic.xlsx')
df_mesos_raw = pd.read_excel(raw_path / 'pib_mesos.xlsx')

df_munic = df_munic_raw.set_index('munic').T

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
