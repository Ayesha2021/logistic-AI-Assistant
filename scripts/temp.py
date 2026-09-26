import pandas as pd
from pathlib import Path

script_dir = Path(__file__).resolve().parent  # points to experiment/phase-0
project_root = script_dir.parents[0]  # climbs up 1 levels to project root
legacy_mapping = {
    'timestamp': 'TS_UTC',
    'vehicle_gps_latitude': 'V_LAT',
    'vehicle_gps_longitude': 'V_LON',
    'iot_temperature': 'IOT_TEMP_VAL_C',
    'cargo_condition_status': 'CGO_COND_CD',
    'risk_classification': 'RISK_CLS_TXT',
    'delay_probability': 'DELAY_PROB_DEC',
    'port_congestion_level': 'PRT_CNG_LVL',
    'route_risk_level': 'RT_RSK_IDX'
}
print(type(legacy_mapping.keys()))
print(legacy_mapping.keys())
data_path = project_root / "data" / "raw" / \
    "dynamic_supply_chain_logistics_dataset.csv"
print(f"Loading CSV from {data_path}...")
df = pd.read_csv(data_path)
df_legacy = df[list(legacy_mapping.keys())].rename(columns=legacy_mapping)
df_legacy['SYS_INGEST_FLAG'] = 'Y'
print(df.head(2))
print(df_legacy.head(2))
