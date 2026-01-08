import polars as pl
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


################# CARREGANDO OS DADOS #################
df_pib_mesos_raw = pl.read_excel(raw_path / 'pib_mesos.xlsx')


################# TRATAMENTO BÁSICO #################
df_pib_mesos_raw = df_pib_mesos_raw.with_columns(
    pl.col("ano").cast(pl.Int32)
    ).set_sorted("ano").drop_nulls()

df_pib_mesos_raw.head()

### EXTRAINDO OS LOGARITMOS E AS DIFERENÇAS
df_pib_mesos = df_pib_mesos_raw.select(
    pl.col("ano"),
    *[pl.col(col).log().alias(col) for col in df_pib_mesos_raw.columns if col != "ano"]
).with_columns(
    [pl.col(col).diff().alias(col) for col in df_pib_mesos_raw.columns if col != "ano"]
).drop_nulls()

df_pib_mesos.head()


################# REGRESSÃO LOG-LOG #################
reg_elast_granflor = sm.glm('pib_granflor ~ pib_sc', data=df_pib_mesos).fit(cov_type='HC0')
reg_elast_oeste = sm.glm('pib_oeste ~ pib_sc', data=df_pib_mesos).fit(cov_type='HC0')
reg_elast_norte = sm.glm('pib_norte ~ pib_sc', data=df_pib_mesos).fit(cov_type='HC0')
reg_elast_serrana = sm.glm('pib_serrana ~ pib_sc', data=df_pib_mesos).fit(cov_type='HC0')
reg_elast_vale = sm.glm('pib_vale ~ pib_sc', data=df_pib_mesos).fit(cov_type='HC0')
reg_elast_sul = sm.glm('pib_sul ~ pib_sc', data=df_pib_mesos).fit(cov_type='HC0')

print(reg_elast_vale.summary())

regs = {
    'pib_granflor': reg_elast_granflor,
    'pib_oeste': reg_elast_oeste,
    'pib_norte': reg_elast_norte,
    'pib_serrana': reg_elast_serrana,
    'pib_vale': reg_elast_vale,
    'pib_sul': reg_elast_sul,
}

### ARMAZENANDO AS ELASTICIDADES EM UM DATAFRAME
elasticidades_df = pd.DataFrame({reg: model.params for reg, model in regs.items()})


################# CARREGANDO AS PROJEÇÕES #################
projs_mesos_raw = pl.read_csv(interim_path / 'pib_sc_projs.csv')

projs_mesos_raw = projs_mesos_raw.select(
    pl.col("ano"),
    pl.col("pib_sc")
)

projs_mesos_raw = projs_mesos_raw.with_columns(
    pl.col("ano").cast(pl.Int32)
    ).set_sorted("ano")

projs_mesos_raw.head()

### EXTRAINDO OS LOGARITMOS E AS DIFERENÇAS
projs_mesos = projs_mesos_raw.select(
    pl.col("ano"),
    *[pl.col(col).log().alias(col) for col in projs_mesos_raw.columns if col != "ano"]
).with_columns(
    [pl.col(col).diff().alias(col) for col in projs_mesos_raw.columns if col != "ano"]
).drop_nulls()

projs_mesos.head()

### CRIA UM DATAFRAME VAZIO PARA AS PROJEÇÕES DO PIB DOS MESORREGIÕES
projs_mesos_pred = pd.DataFrame(index=range(2022, 2031), columns=['pib_sc'])

projs_mesos_pred.head()

proj_mesos_pd = projs_mesos.to_pandas()

proj_mesos_pd = proj_mesos_pd.set_index('ano')

### TRANSFERE OS VALORES PROJETADOS DO PIB DE SC PARA O DATAFRAME DE PROJEÇÕES
for ano in projs_mesos_pred.index:
    projs_mesos_pred.loc[ano, 'pib_sc'] = proj_mesos_pd.loc[ano, 'pib_sc']

projs_mesos_pred = projs_mesos_pred.astype(float)

projs_mesos_pred.head(10)

### APLICA AS PREVISÕES DE CRESCIMENTO DO PIB DE SC NAS REGRESSÕES DAS MESORREGIÕES
for col, reg in regs.items():
    pred = reg.predict(projs_mesos.to_pandas())
    projs_mesos = projs_mesos.with_columns(
        pl.Series(col, np.exp(pred) - 1)
    )

projs_mesos_pd = projs_mesos.to_pandas()

projs_mesos.head()

projs_mesos.write_csv(interim_path / 'pib_mesos_projs.csv')