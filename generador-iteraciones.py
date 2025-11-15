"""
Generador de Instancias para Optimización de Planificación de Personal de Enfermería
Con control de factibilidad basado en condiciones matemáticas
Fundamentos de Investigación de Operaciones
"""
import numpy as np
import pandas as pd
import random
import os
from pathlib import Path

class GeneradorInstanciasFactibles:
    def __init__(self, semilla=52):
        """Inicializa el generador con una semilla para reproducibilidad"""
        random.seed(semilla)
        np.random.seed(semilla)
        
    def calcular_factibilidad(self, disposicion, demanda, num_trabajadores, num_dias, num_turnos):
        """
        Calcula indicadores de factibilidad basados en las condiciones matemáticas
        """
        # Convertir a arrays numpy para cálculos
        disp_array = np.array(disposicion)
        dem_array = np.array(demanda)
        
        # Condición 1: Cobertura básica por turno-día
        trabajadores_disponibles = np.sum(disp_array > 0, axis=0)  # Suma sobre trabajadores
        cobertura_suficiente = trabajadores_disponibles >= dem_array
        
        ratio_cobertura = np.min(trabajadores_disponibles / np.maximum(dem_array, 1))
        
        # Condición 2: Capacidad global del sistema
        demanda_total = np.sum(dem_array)
        capacidad_maxima = 2 * num_trabajadores * num_dias
        ratio_capacidad = capacidad_maxima / max(demanda_total, 1)
        
        # Condición 3: Días críticos
        demanda_por_dia = np.sum(dem_array, axis=1)  # Suma sobre turnos
        capacidad_por_dia = 2 * num_trabajadores
        dias_criticos = np.sum(demanda_por_dia > capacidad_por_dia)
        
        return {
            'ratio_cobertura': ratio_cobertura,
            'ratio_capacidad': ratio_capacidad,
            'dias_criticos': dias_criticos,
            'cobertura_suficiente': np.all(cobertura_suficiente),
            'factible_basico': ratio_cobertura >= 1.0 and ratio_capacidad >= 1.0 and dias_criticos == 0
        }
    
    def generar_disposicion_controlada(self, num_trabajadores, num_dias, num_turnos, densidad=0.85):
        """
        Genera disposición con control de densidad para asegurar factibilidad
        """
        # Matriz base con distribución uniforme
        disposicion = np.random.randint(0, 11, size=(num_trabajadores, num_dias, num_turnos))
        
        # Asegurar densidad mínima de disponibilidad
        mascara_ceros = np.random.random(size=(num_trabajadores, num_dias, num_turnos)) > densidad
        disposicion[mascara_ceros] = 0
        
        return disposicion.tolist()
    
    def generar_demanda_factible(self, num_dias, num_turnos, num_trabajadores, disposicion, nivel_tension="moderada"):
        """
        Genera demanda que cumple condiciones de factibilidad
        """
        # Calcular capacidad disponible por turno-día
        disp_array = np.array(disposicion)
        capacidad_disponible = np.sum(disp_array > 0, axis=0)  # Trabajadores disponibles por (día, turno)
        
        # Capacidad promedio del sistema
        capacidad_promedio = num_trabajadores / num_turnos
        
        # Parámetros según nivel de tensión
        if nivel_tension == "alta_factibilidad":
            mu_factor = 0.6
            sigma_factor = 0.1
            min_ratio = 1.5  # Márgen amplio
        elif nivel_tension == "moderada":
            mu_factor = 0.75
            sigma_factor = 0.2
            min_ratio = 1.2  # Márgen moderado
        else:  # riesgo_infactibilidad
            mu_factor = 0.85
            sigma_factor = 0.3
            min_ratio = 1.0  # Mínimo necesario
        
        demanda = []
        for d in range(num_dias):
            dia_demandas = []
            for t in range(num_turnos):
                # Límite superior basado en disponibilidad real
                limite_superior = min(capacidad_disponible[d, t], capacidad_promedio * 1.5)
                
                # Media ajustada por capacidad
                mu = min(mu_factor * capacidad_promedio, limite_superior * 0.9)
                sigma = sigma_factor * mu
                
                # Generar valor normal truncado
                valor = np.random.normal(mu, sigma)
                valor_truncado = max(1, min(round(valor), limite_superior))
                dia_demandas.append(int(valor_truncado))
            demanda.append(dia_demandas)
        
        return demanda
    
    def ajustar_demanda_para_factibilidad(self, demanda, disposicion, num_trabajadores, num_dias):
        """
        Ajusta la demanda para cumplir condiciones de factibilidad
        """
        dem_array = np.array(demanda)
        disp_array = np.array(disposicion)
        
        # Ajuste 1: Garantizar cobertura por turno-día
        capacidad_disponible = np.sum(disp_array > 0, axis=0)
        dem_array = np.minimum(dem_array, capacidad_disponible)
        
        # Ajuste 2: Garantizar capacidad por día
        demanda_por_dia = np.sum(dem_array, axis=1)
        capacidad_por_dia = 2 * num_trabajadores
        for d in range(num_dias):
            if demanda_por_dia[d] > capacidad_por_dia:
                factor_ajuste = capacidad_por_dia / demanda_por_dia[d]
                dem_array[d, :] = np.floor(dem_array[d, :] * factor_ajuste).astype(int)
        
        # Ajuste 3: Garantizar capacidad global
        demanda_total = np.sum(dem_array)
        capacidad_maxima = 2 * num_trabajadores * num_dias
        if demanda_total > capacidad_maxima:
            factor_ajuste = capacidad_maxima / demanda_total
            dem_array = np.floor(dem_array * factor_ajuste).astype(int)
        
        return dem_array.tolist()
    def forzar_infactibilidad(self, instancia):
        """
        Modifica la demanda de una instancia inicialmente factible
        para volverla infactible respecto a la cobertura básica.
        """
        params = instancia["parametros"]
        num_trabajadores = params["num_trabajadores"]
        num_dias = params["num_dias"]
        num_turnos = len(params["turnos"])
        
        disposicion = np.array(instancia["disposicion"])
        demanda = np.array(instancia["demanda"])
        
        # Capacidad disponible por (día, turno)
        capacidad_disponible = np.sum(disposicion > 0, axis=0)  # shape: (num_dias, num_turnos)
        
        # Elegir aleatoriamente un día y turno donde haya al menos 1 trabajador disponible
        dias_validos, turnos_validos = np.where(capacidad_disponible > 0)
        if len(dias_validos) == 0:
            # En caso extremo de que no haya disponibilidad, simplemente aumentamos en (0,0)
            d, t = 0, 0
            capacidad_local = 0
        else:
            idx = np.random.randint(0, len(dias_validos))
            d = int(dias_validos[idx])
            t = int(turnos_validos[idx])
            capacidad_local = capacidad_disponible[d, t]
        
        # Forzar infactibilidad de cobertura:
        # Demanda mayor que la cantidad de trabajadores disponibles en ese día-turno
        demanda[d, t] = capacidad_local + 1
        
        # Opcional: también podemos inflar un poco la demanda total para bajar ratio_capacidad
        # (no es estrictamente necesario, pero refuerza la infactibilidad)
        demanda_total = np.sum(demanda)
        capacidad_maxima = 2 * num_trabajadores * num_dias
        if demanda_total <= capacidad_maxima:
            # Multiplicamos toda la matriz de demanda para superar la capacidad global
            factor = 1.2
            demanda = np.ceil(demanda * factor).astype(int)
        
        # Actualizar la instancia
        instancia["demanda"] = demanda.tolist()
        instancia["factibilidad"] = self.calcular_factibilidad(
            instancia["disposicion"],
            instancia["demanda"],
            num_trabajadores,
            num_dias,
            num_turnos
        )
        
        return instancia

    def determinar_fines_semana(self, num_dias):
        """
        Determina los conjuntos D_WE(w) para fines de semana
        Considera que el horizonte siempre comienza en lunes
        """
        fines_semana = []
        
        for semana in range((num_dias + 6) // 7):
            sabado = semana * 7 + 6  # Día 6 = sábado (0-indexed)
            domingo = semana * 7 + 7  # Día 7 = domingo (0-indexed)
            
            fin_semana = []
            if sabado < num_dias:
                fin_semana.append(sabado)
            if domingo < num_dias:
                fin_semana.append(domingo)
            
            if fin_semana:
                fines_semana.append([d + 1 for d in fin_semana])  # Convertir a 1-indexed
        
        return fines_semana
    
    def guardar_instancia_csv(self, instancia, carpeta="instancias_factibles"):
        """Guarda todos los archivos CSV para una instancia"""
        os.makedirs(carpeta, exist_ok=True)
        instancia_id = instancia["id"]
        
        # Guardar disposición
        datos_disp = []
        for i in range(instancia['parametros']['num_trabajadores']):
            for d in range(instancia['parametros']['num_dias']):
                for t_idx, turno in enumerate(instancia['parametros']['turnos']):
                    datos_disp.append({
                        'trabajador': f"T{i+1:03d}",
                        'dia': d + 1,
                        'turno': turno,
                        'disposicion': instancia['disposicion'][i][d][t_idx]
                    })
        
        df_disp = pd.DataFrame(datos_disp)
        df_disp.to_csv(f"{carpeta}/{instancia_id}_disposicion.csv", index=False)
        
        # Guardar demanda
        datos_dem = []
        for d in range(instancia['parametros']['num_dias']):
            for t_idx, turno in enumerate(instancia['parametros']['turnos']):
                datos_dem.append({
                    'dia': d + 1,
                    'turno': turno,
                    'demanda': instancia['demanda'][d][t_idx]
                })
        
        df_dem = pd.DataFrame(datos_dem)
        df_dem.to_csv(f"{carpeta}/{instancia_id}_demanda.csv", index=False)
        
        # Guardar parámetros
        datos_params = [
            {'parametro': 'num_trabajadores', 'valor': instancia['parametros']['num_trabajadores']},
            {'parametro': 'num_dias', 'valor': instancia['parametros']['num_dias']},
            {'parametro': 'num_turnos', 'valor': len(instancia['parametros']['turnos'])},
            {'parametro': 'num_fines_semana', 'valor': instancia['parametros']['num_fines_semana']},
            {'parametro': 'nivel_tension', 'valor': instancia['parametros']['nivel_tension']},
            {'parametro': 'ratio_cobertura', 'valor': f"{instancia['factibilidad']['ratio_cobertura']:.3f}"},
            {'parametro': 'ratio_capacidad', 'valor': f"{instancia['factibilidad']['ratio_capacidad']:.3f}"},
            {'parametro': 'factible_basico', 'valor': instancia['factibilidad']['factible_basico']}
        ]
        
        for i, turno in enumerate(instancia['parametros']['turnos']):
            datos_params.append({'parametro': f'turno_{i+1}', 'valor': turno})
        
        for i, fin_semana in enumerate(instancia['parametros']['fines_semana']):
            datos_params.append({'parametro': f'fin_semana_{i+1}', 'valor': ','.join(map(str, fin_semana))})
        
        df_params = pd.DataFrame(datos_params)
        df_params.to_csv(f"{carpeta}/{instancia_id}_parametros.csv", index=False)
        
        return {
            "disposicion": f"{carpeta}/{instancia_id}_disposicion.csv",
            "demanda": f"{carpeta}/{instancia_id}_demanda.csv",
            "parametros": f"{carpeta}/{instancia_id}_parametros.csv"
        }
    
    def generar_instancia_pequena(self, id_instancia):
        """Genera una instancia pequeña con control de factibilidad"""
        num_dias = random.randint(5, 7)
        num_trabajadores = random.randint(8, 15)  # Mínimo aumentado para factibilidad
        turnos = ["Mañana", "Noche"]
        
        # Generar con alta densidad de disponibilidad
        disposicion = self.generar_disposicion_controlada(num_trabajadores, num_dias, len(turnos), densidad=0.9)
        demanda = self.generar_demanda_factible(num_dias, len(turnos), num_trabajadores, disposicion, "alta_factibilidad")
        demanda = self.ajustar_demanda_para_factibilidad(demanda, disposicion, num_trabajadores, num_dias)
        fines_semana = self.determinar_fines_semana(num_dias)
        
        # Calcular factibilidad
        factibilidad = self.calcular_factibilidad(disposicion, demanda, num_trabajadores, num_dias, len(turnos))
        
        instancia = {
            "id": f"pequena_{id_instancia}",
            "tamaño": "pequeña",
            "parametros": {
                "num_trabajadores": num_trabajadores,
                "num_dias": num_dias,
                "turnos": turnos,
                "num_fines_semana": len(fines_semana),
                "fines_semana": fines_semana,
                "nivel_tension": "alta_factibilidad"
            },
            "disposicion": disposicion,
            "demanda": demanda,
            "factibilidad": factibilidad,
            "tipo": "factible"  # NUEVO
        }

        
        return instancia
    
    def generar_instancia_mediana(self, id_instancia):
        """Genera una instancia mediana con control de factibilidad"""
        num_dias = random.randint(7, 14)
        num_trabajadores = random.randint(20, 45)  # Rango aumentado
        turnos = ["Mañana", "Tarde", "Noche"]
        
        niveles = ["moderada", "riesgo_infactibilidad"]
        nivel = random.choice(niveles)
        densidad = 0.85 if nivel == "moderada" else 0.8
        
        disposicion = self.generar_disposicion_controlada(num_trabajadores, num_dias, len(turnos), densidad=densidad)
        demanda = self.generar_demanda_factible(num_dias, len(turnos), num_trabajadores, disposicion, nivel)
        demanda = self.ajustar_demanda_para_factibilidad(demanda, disposicion, num_trabajadores, num_dias)
        fines_semana = self.determinar_fines_semana(num_dias)
        
        factibilidad = self.calcular_factibilidad(disposicion, demanda, num_trabajadores, num_dias, len(turnos))
        
        instancia = {
            "id": f"mediana_{id_instancia}",
            "tamaño": "mediana",
            "parametros": {
                "num_trabajadores": num_trabajadores,
                "num_dias": num_dias,
                "turnos": turnos,
                "num_fines_semana": len(fines_semana),
                "fines_semana": fines_semana,
                "nivel_tension": nivel
            },
            "disposicion": disposicion,
            "demanda": demanda,
            "factibilidad": factibilidad,
            "tipo": "factible"  # NUEVO
        }

        
        return instancia
    
    def generar_instancia_grande(self, id_instancia):
        """Genera una instancia grande con control de factibilidad"""
        num_dias = random.randint(14, 28)
        num_trabajadores = random.randint(50, 90)  # Rango ajustado
        turnos = ["Mañana", "Tarde", "Noche"]
        
        nivel = "riesgo_infactibilidad"
        disposicion = self.generar_disposicion_controlada(num_trabajadores, num_dias, len(turnos), densidad=0.8)
        demanda = self.generar_demanda_factible(num_dias, len(turnos), num_trabajadores, disposicion, nivel)
        demanda = self.ajustar_demanda_para_factibilidad(demanda, disposicion, num_trabajadores, num_dias)
        fines_semana = self.determinar_fines_semana(num_dias)
        
        factibilidad = self.calcular_factibilidad(disposicion, demanda, num_trabajadores, num_dias, len(turnos))
        
        instancia = {
            "id": f"grande_{id_instancia}",
            "tamaño": "grande",
            "parametros": {
                "num_trabajadores": num_trabajadores,
                "num_dias": num_dias,
                "turnos": turnos,
                "num_fines_semana": len(fines_semana),
                "fines_semana": fines_semana,
                "nivel_tension": nivel
            },
            "disposicion": disposicion,
            "demanda": demanda,
            "factibilidad": factibilidad,
            "tipo": "factible"  # NUEVO
        }

        
        return instancia
    
    def generar_conjunto_instancias(self, max_intentos=10, factibles_por_tamaño=3, infactibles_por_tamaño=2):
        """
        Genera, para cada tamaño (pequeña, mediana, grande),
        factibles_por_tamaño instancias factibles y
        infactibles_por_tamaño instancias infactibles.

        Por defecto: 3 factibles y 2 infactibles (total 5 por tamaño).
        """
        instancias = []
        
        print("Generando instancias pequeñas...")
        # --- FACTIBLES ---
        for i in range(1, factibles_por_tamaño + 1):
            for intento in range(max_intentos):
                instancia = self.generar_instancia_pequena(i)
                if instancia['factibilidad']['factible_basico']:
                    instancias.append(instancia)
                    print(f"  ✔ {instancia['id']} (factible): RC={instancia['factibilidad']['ratio_cobertura']:.2f}")
                    break
                elif intento == max_intentos - 1:
                    print(f"  ⚠ {instancia['id']}: usando instancia aunque no cumpla todas las condiciones")
                    instancias.append(instancia)
        
        # --- INFACTIBLES ---
        for i in range(1, infactibles_por_tamaño + 1):
            instancia = self.generar_instancia_pequena(i + factibles_por_tamaño)
            instancia_infact = self.forzar_infactibilidad(instancia)
            instancias.append(instancia_infact)
            print(f"  ✖ {instancia_infact['id']} (infactible): "
                  f"RC={instancia_infact['factibilidad']['ratio_cobertura']:.2f}, "
                  f"FB={instancia_infact['factibilidad']['factible_basico']}")
        
        print("Generando instancias medianas...")
        # --- FACTIBLES ---
        for i in range(1, factibles_por_tamaño + 1):
            for intento in range(max_intentos):
                instancia = self.generar_instancia_mediana(i)
                if instancia['factibilidad']['factible_basico']:
                    instancias.append(instancia)
                    print(f"  ✔ {instancia['id']} (factible): RC={instancia['factibilidad']['ratio_cobertura']:.2f}")
                    break
                elif intento == max_intentos - 1:
                    print(f"  ⚠ {instancia['id']}: usando instancia aunque no cumpla todas las condiciones")
                    instancias.append(instancia)
        
        # --- INFACTIBLES ---
        for i in range(1, infactibles_por_tamaño + 1):
            instancia = self.generar_instancia_mediana(i + factibles_por_tamaño)
            instancia_infact = self.forzar_infactibilidad(instancia)
            instancias.append(instancia_infact)
            print(f"  ✖ {instancia_infact['id']} (infactible): "
                  f"RC={instancia_infact['factibilidad']['ratio_cobertura']:.2f}, "
                  f"FB={instancia_infact['factibilidad']['factible_basico']}")
        
        print("Generando instancias grandes...")
        # --- FACTIBLES ---
        for i in range(1, factibles_por_tamaño + 1):
            for intento in range(max_intentos):
                instancia = self.generar_instancia_grande(i)
                # Para grandes usabas una condición un poco más relajada
                if instancia['factibilidad']['cobertura_suficiente']:
                    instancias.append(instancia)
                    print(f"  ✔ {instancia['id']} (factible seg. cobertura): RC={instancia['factibilidad']['ratio_cobertura']:.2f}")
                    break
                elif intento == max_intentos - 1:
                    print(f"  ⚠ {instancia['id']}: usando instancia disponible")
                    instancias.append(instancia)
        
        # --- INFACTIBLES ---
        for i in range(1, infactibles_por_tamaño + 1):
            instancia = self.generar_instancia_grande(i + factibles_por_tamaño)
            instancia_infact = self.forzar_infactibilidad(instancia)
            instancias.append(instancia_infact)
            print(f"  ✖ {instancia_infact['id']} (infactible): "
                  f"RC={instancia_infact['factibilidad']['ratio_cobertura']:.2f}, "
                  f"FB={instancia_infact['factibilidad']['factible_basico']}")
        
        return instancias
    
    def crear_resumen_csv(self, instancias, carpeta="instancias_factibles"):
        """Crea un archivo CSV con resumen de todas las instancias"""
        datos_resumen = []
        
        for instancia in instancias:
            params = instancia["parametros"]
            fact = instancia["factibilidad"]
            
            # Calcular estadísticas de demanda
            todas_demandas = [dem for dia in instancia["demanda"] for dem in dia]
            demanda_promedio = np.mean(todas_demandas)
            demanda_maxima = np.max(todas_demandas)
            
            datos_resumen.append({
                'instancia_id': instancia['id'],
                'tamaño': instancia['tamaño'],
                'num_trabajadores': params['num_trabajadores'],
                'num_dias': params['num_dias'],
                'num_turnos': len(params['turnos']),
                'num_fines_semana': params['num_fines_semana'],
                'nivel_tension': params['nivel_tension'],
                'ratio_cobertura': fact['ratio_cobertura'],
                'ratio_capacidad': fact['ratio_capacidad'],
                'factible_basico': fact['factible_basico'],
                'cobertura_suficiente': fact['cobertura_suficiente'],
                'dias_criticos': fact['dias_criticos'],
                'demanda_promedio': round(demanda_promedio, 2),
                'demanda_maxima': demanda_maxima,
                'turnos': '|'.join(params['turnos'])
            })
        
        df_resumen = pd.DataFrame(datos_resumen)
        archivo_resumen = f"{carpeta}/resumen_instancias.csv"
        df_resumen.to_csv(archivo_resumen, index=False)
        
        return archivo_resumen
    

def main():
    """Función principal del generador"""
    print("GENERADOR DE INSTANCIAS FACTIBLES - PLANIFICACIÓN DE ENFERMERÍA")
    print("CON CONTROL DE FACTIBILIDAD MATEMÁTICA")
    print("="*70)
    
    # Crear generador
    generador = GeneradorInstanciasFactibles(semilla=12345)
    
    # Generar instancias
    instancias = generador.generar_conjunto_instancias()
    
    # Guardar cada instancia en archivos CSV
    print("\nGuardando archivos CSV...")
    archivos_creados = []
    
    for instancia in instancias:
        print(f"  - Procesando {instancia['id']}...")
        archivos = generador.guardar_instancia_csv(instancia)
        archivos_creados.append(archivos)
    
    # Crear resumen
    archivo_resumen = generador.crear_resumen_csv(instancias)
    
    # Mostrar estadísticas finales
    print("\n" + "="*60)
    print("RESUMEN FINAL DE FACTIBILIDAD")
    print("="*60)
    
    factibles = sum(1 for inst in instancias if inst['factibilidad']['factible_basico'])
    cobertura_ok = sum(1 for inst in instancias if inst['factibilidad']['cobertura_suficiente'])
    
    print(f"Instancias generadas: {len(instancias)}")
    print(f"Instancias factibles (básico): {factibles}/{len(instancias)}")
    print(f"Instancias con cobertura suficiente: {cobertura_ok}/{len(instancias)}")
    print(f"📁 Carpeta de destino: instancias_factibles/")
    print(f"📊 Archivo de resumen: {os.path.basename(archivo_resumen)}")

if __name__ == "__main__":
    main()