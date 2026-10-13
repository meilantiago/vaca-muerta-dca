import pandas as pd
import glob
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_sesco_data(raw_dir: str, out_dir: str):
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(raw_path.glob("produccion_*.csv"))
    if not csv_files:
        logging.error(f"No CSV files found in {raw_dir}")
        return
        
    logging.info(f"Found {len(csv_files)} files to process.")
    
    # Columnas que realmente nos importan para el análisis
    usecols = [
        'idpozo', 'sigla', 'empresa', 'areayacimiento', 'cuenca', 'provincia', 
        'formacion', 'tipo_de_recurso', 'sub_tipo_recurso', 
        'fecha_data', 'prod_pet', 'prod_gas', 'prod_agua', 'tef'
    ]
    
    # Diccionario para forzar tipos y ahorrar memoria / evitar warnings
    dtype_dict = {
        'idpozo': 'int32',
        'prod_pet': 'float32',
        'prod_gas': 'float32',
        'prod_agua': 'float32',
        'tef': 'float32'
    }

    dfs = []
    for file in csv_files:
        logging.info(f"Loading {file.name}...")
        try:
            # chunksize si la memoria fuera un problema, pero 300MB entra en RAM
            df = pd.read_csv(file, usecols=usecols, dtype=dtype_dict, low_memory=False)
            dfs.append(df)
        except Exception as e:
            logging.error(f"Error loading {file.name}: {e}")
            
    df_full = pd.concat(dfs, ignore_index=True)
    logging.info(f"Total rows loaded: {len(df_full):,}")
    
    # 1. Filtrar Vaca Muerta
    logging.info("Filtering for Vaca Muerta...")
    vm_mask = (df_full['cuenca'].str.upper() == 'NEUQUINA') & \
              (df_full['formacion'].str.upper().str.contains('VACA MUERTA', na=False))
    
    df_vm = df_full[vm_mask].copy()
    logging.info(f"Rows after filtering Vaca Muerta: {len(df_vm):,}")
    
    # 2. Limpieza de fechas y ordenamiento
    logging.info("Processing dates and sorting...")
    df_vm['fecha_data'] = pd.to_datetime(df_vm['fecha_data'], errors='coerce')
    df_vm = df_vm.dropna(subset=['fecha_data'])
    df_vm = df_vm.sort_values(by=['idpozo', 'fecha_data']).reset_index(drop=True)
    
    # 3. Filtrar datos anómalos (ej: TEF negativo o producción negativa)
    df_vm = df_vm[(df_vm['tef'] >= 0) & (df_vm['prod_pet'] >= 0) & (df_vm['prod_gas'] >= 0)]
    
    # 4. Calcular el "mes relativo" (mes 1, mes 2, etc. desde el inicio de producción del pozo)
    # Definimos inicio de producción como el primer mes donde tef > 0 o prod_pet > 0 o prod_gas > 0
    logging.info("Calculating relative production months...")
    
    # Encontrar la primera fecha de producción real por pozo
    prod_activa = df_vm[(df_vm['tef'] > 0) | (df_vm['prod_pet'] > 0) | (df_vm['prod_gas'] > 0)]
    inicio_prod = prod_activa.groupby('idpozo')['fecha_data'].min().rename('fecha_inicio')
    
    # Joinear la fecha de inicio al dataframe principal
    df_vm = df_vm.merge(inicio_prod, on='idpozo', how='left')
    
    # Descartar registros anteriores al inicio de producción real (pozos en perforación/terminación)
    df_vm = df_vm[df_vm['fecha_data'] >= df_vm['fecha_inicio']].copy()
    
    # Calcular meses relativos: (año_actual - año_inicio)*12 + (mes_actual - mes_inicio) + 1
    df_vm['mes_relativo'] = (df_vm['fecha_data'].dt.year - df_vm['fecha_inicio'].dt.year) * 12 + \
                            (df_vm['fecha_data'].dt.month - df_vm['fecha_inicio'].dt.month) + 1
                            
    # 5. Calcular producción acumulada
    df_vm['pet_acumulado'] = df_vm.groupby('idpozo')['prod_pet'].cumsum()
    df_vm['gas_acumulado'] = df_vm.groupby('idpozo')['prod_gas'].cumsum()
    
    # 6. Descartar pozos con muy pocos datos para curvas (ej: menos de 3 meses)
    # Nota: para este preprocesamiento lo dejamos, el filtro se hace en el modelo
    
    out_file = out_path / "vaca_muerta_prod.parquet"
    logging.info(f"Saving processed data to {out_file}...")
    df_vm.to_parquet(out_file, engine='pyarrow', index=False)
    
    logging.info(f"Done! Processed {len(df_vm['idpozo'].unique())} unique wells.")
    
if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    raw_dir = project_root / "data" / "raw"
    out_dir = project_root / "data" / "processed"
    
    process_sesco_data(raw_dir, out_dir)
