# ------------------------------------
# SCRIPT PARA PROJEÇÃO DO PIB DE SANTA CATARINA ATÉ 2030
# Modelo simples com base nas elasticidades em relação ao
# PIB do Brasil, EUA, China, Selic e Câmbio.
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
# CARREGAMENTO DOS DADOS
# ------------------------------------
df_pib_sc_raw = pl.read_excel(raw_path / 'pib_sc.xlsx')


# ------------------------------------
# TRATAMENTO E DEFINIÇÃO DAS VARIÁVEIS
# ------------------------------------
df_pib_sc_raw = df_pib_sc_raw.with_columns(
    pl.col("ano").cast(pl.Int32),
    pl.col('selic_over').shift(1).alias('selic_over_1'),
    pl.col('cambio_uss').shift(1).alias('cambio_uss_1')
    ).set_sorted("ano").drop_nulls()

df_pib_sc_raw.head()


# ------------------------------------
# REGRESSÃO LOG-LOG PARA CÁLCULO DAS ELASTICIDADES
# ------------------------------------
df_pib_sc_pd = df_pib_sc_raw.to_pandas()

expr = """
np.log(pib_sc) ~ np.log(pib_br) + np.log(pib_usa)
               + np.log(pib_chn) + selic_over_1 + np.log(cambio_uss_1)
"""

reg_elast = sm.glm(expr, data=df_pib_sc_pd).fit(cov_type='HC0')

print(reg_elast.summary())


'''
CENÁRIO-BASE 05-2025:
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

# ------------------------------------
# Projeção das variáveis explicativas até 2030
# Aqui são definidas as taxas de crescimento reais para cada variável.
# É bem manual, com coleta das projeções nas fontes citadas acima.
# São três cenários: base, pessimista e otimista.
# Os cenários para câmbio e Selic são definidos na sequência.
# ------------------------------------
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

ultimo_ano = df_pib_sc_raw.select(pl.col("ano")).to_series()[-1]

valores_atuais = df_pib_sc_raw.filter(pl.col("ano") == ultimo_ano).select(['pib_br', 'pib_usa', 'pib_chn']).to_numpy()[0]

# ------------------------------------
# SELEÇÃO DO CENÁRIO BASE PARA AS PROJEÇÕES DAS VARIÁVEIS EXPLICATIVAS
# ------------------------------------
taxas = taxas_base.copy()

### CRIA UM DATAFRAME VAZIO PARA ARMAZENAR AS PROJEÇÕES
projs_sc = pd.DataFrame(index=taxas.index, columns=['pib_br', 'pib_usa', 'pib_chn'])

### CALCULA OS VALORES FUTUROS COM BASE NAS TAXAS DE CRESCIMENTO
for ano in taxas.index:
    crescimento = taxas.loc[ano]
    valores_atuais = valores_atuais*(1 + crescimento)
    projs_sc.loc[ano] = valores_atuais

projs_sc = projs_sc.astype(float)

### DEFINE OS CENÁRIOS PARA A TAXA SELIC
selic_base = [10.83, 14, 12.5, 10.5, 10.0, 10.0]
selic_otim = [10.83, 12.5, 11.5, 10.5, 8, 8]
selic_pess = [10.83, 14.5, 13, 12, 10.0, 10.0] 

### AQUI SE SELECIONA O CENÁRIO DE PROJEÇÃO TAXA SELIC
projs_sc.loc[2025:, 'selic_over_1'] = selic_base

### DEFINE OS CENÁRIOS PARA O CÂMBIO
projs_sc.loc[2025:, 'cambio_uss_1'] = [5.91, 5.85, 5.9, 5.8, 5.82, 5.8]

### FAZ O PREDICT DO PIB DE SC COM BASE NAS PROJEÇÕES DAS VARIÁVEIS INDEPENDENTES
projs_sc['pib_sc'] = np.exp(reg_elast.predict(projs_sc))

projs_sc.head()

# ------------------------------------
# CONCATENAÇÃO E EXPORTAÇÃO DOS DADOS
# ------------------------------------
df_pib_sc_pd = df_pib_sc_pd.set_index('ano')
df_concat_sc = pd.concat([df_pib_sc_pd, projs_sc])

### CÁLCULO DAS TAXAS REAIS DE CRESCIMENTO ANUAL
df_concat_sc_taxas = df_concat_sc.pct_change(fill_method=None)

df_concat_sc.to_csv(interim_path / 'pib_sc_projs.csv', index=True)
