
from turtle import pd
import pyreadr


df_ips_data_2 = pd.read_csv(
    'ispra-rapporto-408_2025.csv',
    delimiter=';', 
    decimal=',', 
    encoding='cp1252'
    )
 
 ## Read coeficients for IPS calculation
result = pyreadr.read_r("ips.rda")
ips_index_values = result["ips"]
ips_index_values.head()

new_rows = (
    df_ips_data_2[["Name", "PS s", "PS v"]]
    .rename(columns={
        "Name": "fullspecies",
        "PS s": "ips_s",
        "PS v": "ips_v"
    })
)

ips_index_values = pd.concat(
    [ips_index_values, new_rows],
    ignore_index=True
)

species_col = "fullspecies"

values = (
    ips_index_values[species_col]
    .astype("string")
    .str.strip()
)

has_suffix = values.str.contains(
    r"\s+(species|sp\.?)$",
    case=False,
    regex=True,
    na=False
)

base_names = (
    values.loc[has_suffix]
    .str.replace(
        r"\s+(species|sp\.?)$",
        "",
        case=False,
        regex=True
    )
    .str.strip()
)

variants = []

for suffix in ["species", "sp", "sp."]:
    variant_rows = ips_index_values.loc[has_suffix].copy()
    variant_rows[species_col] = (
        base_names + " " + suffix
    ).values
    variants.append(variant_rows)

ips_index_values = pd.concat(
    [ips_index_values, *variants],
    ignore_index=True
).drop_duplicates()

ips_index_values.to_csv(
    "IPS_index_values.csv", index=False, sep=",", decimal=".", encoding="utf-8-sig"
)
