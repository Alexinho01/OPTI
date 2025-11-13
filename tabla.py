import pandas as pd
import re

def procesar_solucion_lpsolve(archivo_csv="eralasalsa.csv"):
    """
    Procesa la solución de LPSolve y la convierte en una tabla de asignaciones legible
    """
    
    # === Paso 1: Leer el archivo CSV exportado desde LPsolve IDE ===
    try:
        df = pd.read_csv(archivo_csv, sep=";", skiprows=2, names=["Variable", "MILP", "Result"])
        print(f"✓ Archivo {archivo_csv} leído correctamente")
        print(f"  Total de variables: {len(df)}")
    except Exception as e:
        print(f"✗ Error al leer el archivo: {e}")
        return None
    
    # === Paso 2: Filtrar variables con resultado igual a 1 ===
    df_filtrado = df[df["Result"] == 1]
    print(f"✓ Variables asignadas (Result=1): {len(df_filtrado)}")
    
    # === Paso 3: Extraer información de las variables ===
    asignaciones = []
    
    for var in df_filtrado["Variable"]:
        var_str = str(var).strip()
        
        # Patrón para variables x_i_d_t (ej: x_1_5_Manana, x_3_12_Tarde)
        match_x = re.match(r"x_(\d+)_(\d+)_([A-Za-z]+)", var_str)
        if match_x:
            i, d, t = match_x.groups()
            trabajador = f"T{int(i):03d}"  # Formato T001, T002, etc.
            dia = int(d)
            turno = t
            asignaciones.append({
                'trabajador': trabajador,
                'dia': dia,
                'turno': turno,
                'variable': var_str
            })
    
    print(f"✓ Asignaciones de turnos extraídas: {len(asignaciones)}")
    
    if not asignaciones:
        print("✗ No se encontraron asignaciones de turnos")
        return None
    
    # === Paso 4: Crear tabla en formato horizontal como la imagen ===
    trabajadores = sorted(set(a['trabajador'] for a in asignaciones))
    dias = sorted(set(a['dia'] for a in asignaciones))
    turnos = sorted(set(a['turno'] for a in asignaciones))
    
    print(f"✓ Trabajadores: {len(trabajadores)}")
    print(f"✓ Días: {len(dias)}")
    print(f"✓ Turnos encontrados: {turnos}")
    
    # Crear columnas: D01_Manana, D01_Noche, D02_Manana, D02_Noche, etc.
    columnas = []
    for dia in dias:
        for turno in turnos:
            columna = f"D{dia:02d}_{turno}"
            columnas.append(columna)
    
    # Crear DataFrame vacío
    tabla = pd.DataFrame("", index=trabajadores, columns=columnas)
    
    # Llenar la tabla con las asignaciones
    for asignacion in asignaciones:
        trabajador = asignacion['trabajador']
        dia = asignacion['dia']
        turno = asignacion['turno']
        columna = f"D{dia:02d}_{turno}"
        
        tabla.loc[trabajador, columna] = "✓"
    
    # === Paso 5: Mostrar estadísticas ===
    print("\n" + "="*60)
    print("ESTADÍSTICAS DE ASIGNACIÓN")
    print("="*60)
    
    # Asignaciones por trabajador
    asignaciones_por_trabajador = tabla.apply(lambda x: (x == "✓").sum(), axis=1)
    print(f"\nAsignaciones por trabajador:")
    print(f"  Mínimo: {asignaciones_por_trabajador.min()}")
    print(f"  Máximo: {asignaciones_por_trabajador.max()}")
    print(f"  Promedio: {asignaciones_por_trabajador.mean():.1f}")
    
    # Cobertura por turno
    print(f"\nCobertura por turno:")
    for turno in turnos:
        columnas_turno = [col for col in tabla.columns if col.endswith(f"_{turno}")]
        cobertura_turno = (tabla[columnas_turno] == "✓").sum().sum()
        print(f"  {turno}: {cobertura_turno} asignaciones")
    
    # === Paso 6: Mostrar tabla en formato ordenado ===
    print("\n" + "="*60)
    print("TABLA DE ASIGNACIONES")
    print("="*60)
    
    # Reorganizar columnas para mejor visualización
    columnas_ordenadas = []
    for dia in dias:
        for turno in turnos:
            columna = f"D{dia:02d}_{turno}"
            if columna in tabla.columns:
                columnas_ordenadas.append(columna)
    
    tabla_ordenada = tabla[columnas_ordenadas]
    
    # Mostrar tabla
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(tabla_ordenada.fillna(""))
    
    # === Paso 7: Exportar resultados ===
    # Exportar tabla completa
    tabla_ordenada.fillna("").to_csv("asignaciones_completas.csv")
    print(f"\n✓ Tabla completa exportada: asignaciones_completas.csv")
    
    return tabla_ordenada

def generar_tabla_formato_imagen(tabla, archivo_salida="tabla_formato_imagen.txt"):
    """Genera una tabla en formato similar a la imagen proporcionada"""
    
    trabajadores = tabla.index.tolist()
    columnas = tabla.columns.tolist()
    
    # Extraer días y turnos únicos
    dias = sorted(set(col.split('_')[0] for col in columnas))
    turnos = sorted(set(col.split('_')[1] for col in columnas))
    
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        # Escribir encabezado de días
        f.write("Día    ")
        for dia in dias:
            f.write(f"{dia}    ")
        f.write("\n")
        
        # Escribir encabezado de turnos
        f.write("Turno ")
        for dia in dias:
            for turno in turnos:
                # Abreviar nombres largos
                abrev = turno[:3] if len(turno) > 3 else turno
                f.write(f"{abrev:<6}")
        f.write("\n\n")
        
        # Escribir asignaciones por trabajador
        for trabajador in trabajadores:
            f.write(f"{trabajador} ")
            for dia in dias:
                for turno in turnos:
                    columna = f"{dia}_{turno}"
                    if columna in tabla.columns and tabla.loc[trabajador, columna] == "✓":
                        f.write("  ✓   ")
                    else:
                        f.write("      ")
            f.write("\n")
    
    print(f"✓ Tabla en formato imagen exportada: {archivo_salida}")

# === EJECUCIÓN PRINCIPAL ===
if __name__ == "__main__":
    print("PROCESADOR DE SOLUCIONES LPSOLVE - PLANIFICACIÓN DE ENFERMERÍA")
    print("=" * 70)
    
    # Procesar el archivo CSV de LPSolve
    tabla_asignaciones = procesar_solucion_lpsolve("eralasalsa.csv")
    
    if tabla_asignaciones is not None:
        # Generar tabla en formato similar a la imagen
        generar_tabla_formato_imagen(tabla_asignaciones)
        
        print("\n" + "="*70)
        print("¡PROCESAMIENTO COMPLETADO!")
        print("Archivos generados:")
        print("  - asignaciones_completas.csv (tabla CSV)")
        print("  - tabla_formato_imagen.txt (formato visual)")
    else:
        print("\nNo se pudo procesar la solución. Verifica el archivo de entrada.")