import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time
import os
import traceback # Para imprimir errores detallados
from flask import Flask, request, jsonify

# --- Configuración ---
DIRECCION_DATOS = 'Datos' # Directorio que contiene los archivos CSV
DIRECCION_B2C_TRANSACCIONES = os.path.join(DIRECCION_DATOS, 'transacciones_con_features.csv')
FLASK_PORT = 5002 # Puerto cambiado para evitar conflictos si el otro servidor está corriendo

# --- Variables Globales para Datos Pre-calculados ---
user_item_matrix_csr = None
item_similarity_matrix = None
user_to_idx = None
idx_to_user = None
item_to_idx = None
idx_to_item = None
n_users = 0
n_items = 0
data_ready = False # Bandera para indicar si la configuración fue exitosa

# --- Función para Cargar y Preparar Datos (Adaptada del script anterior) ---
def setup_recommendation_data():
    """Carga datos y calcula matrices UNA VEZ al inicio."""
    global user_item_matrix_csr, item_similarity_matrix
    global user_to_idx, idx_to_user, item_to_idx, idx_to_item
    global n_users, n_items, data_ready

    print("--- Configurando Datos para Recomendación Basada en Ítems ---")
    overall_start_time = time.time()

    # --- 1. Cargar y Preparar Datos ---
    print("Paso 1: Cargando Datos de Transacciones...")
    start_time = time.time()
    try:
        # Verificar si el archivo existe antes de intentar leerlo
        if not os.path.exists(DIRECCION_B2C_TRANSACCIONES):
             raise FileNotFoundError(f"Archivo de transacciones no encontrado: {DIRECCION_B2C_TRANSACCIONES}")

        # Leer solo las columnas necesarias
        transacciones = pd.read_csv(
            DIRECCION_B2C_TRANSACCIONES,
            encoding='utf-8',
            usecols=['id', 'producto']
        )
        # Renombrar columna 'id' a 'cliente_id' por claridad
        transacciones.rename(columns={'id': 'cliente_id'}, inplace=True)
        # Eliminar filas con valores faltantes en columnas clave
        transacciones.dropna(subset=['cliente_id', 'producto'], inplace=True)
        # Convertir cliente_id a numérico (entero), manejando errores
        transacciones['cliente_id'] = pd.to_numeric(transacciones['cliente_id'], errors='coerce')
        transacciones.dropna(subset=['cliente_id'], inplace=True) # Eliminar filas donde la conversión falló
        transacciones['cliente_id'] = transacciones['cliente_id'].astype(int) # Convertir a entero estándar
        # Eliminar duplicados (si un usuario compró el mismo producto varias veces, solo cuenta como una interacción)
        transacciones = transacciones.drop_duplicates(subset=['cliente_id', 'producto'])
        print(f"Cargadas {len(transacciones)} interacciones únicas usuario-producto.")
        print(f"Paso 1 finalizado en {time.time() - start_time:.2f} segundos.")
    except Exception as e:
        print(f"ERROR CRÍTICO en Paso 1 (Cargar Datos): {e}")
        traceback.print_exc() # Imprimir el traceback completo del error
        data_ready = False # Marcar que la configuración falló
        return # Detener la configuración si hay un error crítico

    # --- 2. Crear Mapeos ---
    print("Paso 2: Creando Mapeos de Usuario y Producto...")
    start_time = time.time()
    try:
        # Obtener usuarios y productos únicos y ordenarlos para mapeos consistentes
        unique_clientes = sorted(transacciones['cliente_id'].unique())
        unique_productos = sorted(transacciones['producto'].unique())
        n_users = len(unique_clientes)
        n_items = len(unique_productos)

        # Verificar que se encontraron usuarios e ítems
        if n_users == 0 or n_items == 0:
             print("ERROR CRÍTICO: No se encontraron usuarios o ítems después de procesar los datos.")
             data_ready = False
             return

        # Crear diccionarios para mapear IDs a índices y viceversa
        user_to_idx = {user_id: i for i, user_id in enumerate(unique_clientes)}
        idx_to_user = {i: user_id for user_id, i in user_to_idx.items()}
        item_to_idx = {item_id: i for i, item_id in enumerate(unique_productos)}
        idx_to_item = {i: item_id for item_id, i in item_to_idx.items()}
        print(f"Encontrados {n_users} usuarios únicos y {n_items} ítems únicos.")
        print(f"Paso 2 finalizado en {time.time() - start_time:.2f} segundos.")
    except Exception as e:
        print(f"ERROR CRÍTICO en Paso 2 (Mapeos): {e}")
        traceback.print_exc()
        data_ready = False
        return

    # --- 3. Construir Matriz de Interacción Usuario-Ítem ---
    print("Paso 3: Construyendo Matriz de Interacción Usuario-Ítem...")
    start_time = time.time()
    try:
        temp_user_item_matrix_csr = None # Variable temporal para la matriz
        # Usar lil_matrix para construcción eficiente (asignación de elementos individuales)
        user_item_matrix_lil = lil_matrix((n_users, n_items), dtype=np.int8) # int8 para ahorrar memoria (0 o 1)
        # Iterar sobre las transacciones únicas para llenar la matriz
        for _, row in transacciones.iterrows():
            user_id = row['cliente_id']
            item_id = row['producto']
            # Verificar si los IDs existen en los mapeos (importante si los datos cambian)
            if user_id in user_to_idx and item_id in item_to_idx:
                user_idx = user_to_idx[user_id]
                item_idx = item_to_idx[item_id]
                user_item_matrix_lil[user_idx, item_idx] = 1 # Marcar interacción con 1
        # Convertir a CSR (Compressed Sparse Row) para cálculos más rápidos
        temp_user_item_matrix_csr = user_item_matrix_lil.tocsr()
        del user_item_matrix_lil # Liberar memoria
        del transacciones # Liberar memoria del DataFrame de transacciones
        user_item_matrix_csr = temp_user_item_matrix_csr # Asignar a la variable global

        # Calcular densidad de la matriz (qué porcentaje de celdas no son cero)
        density = user_item_matrix_csr.nnz / (n_users * n_items) if (n_users * n_items) > 0 else 0
        print(f"Matriz Usuario-Ítem construida ({user_item_matrix_csr.shape}), Densidad: {density:.6f}")
        print(f"Paso 3 finalizado en {time.time() - start_time:.2f} segundos.")
    except Exception as e:
        print(f"ERROR CRÍTICO en Paso 3 (Matriz Usuario-Ítem): {e}")
        traceback.print_exc()
        user_item_matrix_csr = None # Asegurar que sea None en caso de error
        data_ready = False
        return

    # --- 4. Calcular Matriz de Similitud Ítem-Ítem ---
    print("Paso 4: Calculando Matriz de Similitud Ítem-Ítem...")
    start_time = time.time()
    # Solo calcular si la matriz Usuario-Ítem se creó correctamente y tiene datos
    if user_item_matrix_csr is None or user_item_matrix_csr.nnz == 0:
        print("Advertencia: Omitiendo cálculo de Similitud Ítem-Ítem (Matriz Usuario-Ítem vacía/inválida).")
        item_similarity_matrix = None
    else:
        try:
            # Transponer la matriz para tener ítems como filas (ítems x usuarios)
            item_user_matrix_csr = user_item_matrix_csr.T.tocsr()
            # Calcular similitud coseno entre ítems (columnas de la matriz original)
            # Usar dense_output=True para obtener una matriz densa (numpy array), más fácil de indexar
            temp_item_similarity_matrix = cosine_similarity(item_user_matrix_csr, dense_output=True)
            # Poner la diagonal a 0 (la similitud de un ítem consigo mismo no es útil para recomendar *otros*)
            np.fill_diagonal(temp_item_similarity_matrix, 0)
            item_similarity_matrix = temp_item_similarity_matrix # Asignar a la variable global
            print(f"Matriz de similitud Ítem-Ítem calculada ({item_similarity_matrix.shape}).")
            print(f"Paso 4 finalizado en {time.time() - start_time:.2f} segundos.")
        except Exception as e:
            print(f"ERROR CRÍTICO en Paso 4 (Similitud Ítem-Ítem): {e}")
            traceback.print_exc()
            item_similarity_matrix = None # Asegurar que sea None en caso de error

    # --- Verificación Final del Setup ---
    # Comprobar si los componentes esenciales están listos
    if user_item_matrix_csr is not None and item_similarity_matrix is not None and user_to_idx and item_to_idx:
        data_ready = True # Marcar como listo si todo está bien
        print(f"--- Configuración Ítem-Based COMPLETADA en {time.time() - overall_start_time:.2f} segundos. Datos listos: {data_ready} ---")
    else:
        data_ready = False # Marcar como no listo si algo falló
        print(f"--- Configuración Ítem-Based FALLIDA después de {time.time() - overall_start_time:.2f} segundos. Datos listos: {data_ready} ---")


# --- Función de Recomendación (Copiada del script anterior - Basada en Ítems) ---
def get_item_based_recommendations_allow_repurchase(target_cliente_id, N=10, k_similar_items=30):
    """
    Genera recomendaciones de productos basadas en la similitud de ítems con el historial de compras,
    PERMITIENDO que ítems ya comprados sean recomendados nuevamente.
    (Asume que las variables globales user_item_matrix_csr, item_similarity_matrix, etc. están configuradas)
    """
    recommendations = pd.DataFrame() # Inicializar DataFrame vacío

    # --- Validación de Entrada y Estado ---
    if not data_ready: # Comprobar si el setup inicial fue exitoso
        print("Error: Los datos de recomendación no están listos.")
        # En una app real, podría devolver 503 Service Unavailable aquí
        return recommendations
    if target_cliente_id not in user_to_idx: # Comprobar si el cliente existe en el mapeo
        print(f"Error: Cliente ID '{target_cliente_id}' no encontrado en el mapeo de usuarios.")
        return recommendations # O devolver 404 Not Found
    if item_similarity_matrix is None: # Comprobar si la matriz de similitud existe
         print("Error: La matriz de similitud Ítem-Ítem no está disponible para recomendaciones.")
         return recommendations # O devolver 503

    # Obtener el índice interno del usuario
    target_user_idx = user_to_idx[target_cliente_id]

    # --- Obtener Historial de Compras del Usuario ---
    try:
        # Verificar que el índice del usuario sea válido para la matriz
        if target_user_idx >= user_item_matrix_csr.shape[0]:
             print(f"Error: Índice de usuario {target_user_idx} fuera de los límites de la matriz Usuario-Ítem.")
             return recommendations
        # Obtener los índices de los ítems comprados por el usuario (de la fila correspondiente en la matriz CSR)
        user_purchased_items_indices = user_item_matrix_csr[target_user_idx].indices
        # Si el historial está vacío, no se pueden generar recomendaciones basadas en él
        if len(user_purchased_items_indices) == 0:
            print(f"El usuario {target_cliente_id} no tiene historial de compras en los datos.")
            return recommendations
    except Exception as e:
        print(f"Error al recuperar el historial de compras para el usuario {target_cliente_id}: {e}")
        traceback.print_exc()
        return recommendations

    # --- Generar Recomendaciones Candidatas ---
    # Usar un defaultdict para acumular puntajes para cada ítem candidato
    candidate_items = defaultdict(float) # índice_ítem -> puntaje_acumulado
    try:
        # Iterar sobre cada ítem que el usuario ha comprado
        for purchased_item_idx in user_purchased_items_indices:
             # Verificar que el índice del ítem comprado sea válido para la matriz de similitud
            if purchased_item_idx >= item_similarity_matrix.shape[0]: continue # Omitir este ítem si su índice no es válido

            # Obtener los puntajes de similitud de este ítem comprado con todos los demás ítems
            item_similarities = item_similarity_matrix[purchased_item_idx]

            # Obtener los índices de los 'k_similar_items' ítems más similares
            # np.argsort devuelve los índices que ordenarían el array; [::-1] los invierte (descendente)
            similar_item_indices = np.argsort(item_similarities)[::-1][:k_similar_items]

            # Iterar sobre los ítems más similares encontrados
            for similar_item_idx in similar_item_indices:
                similarity_score = item_similarities[similar_item_idx] # Obtener el puntaje de similitud

                # Opcional: Considerar solo ítems con similitud positiva
                if similarity_score <= 0:
                    continue

                # *** EL CAMBIO CLAVE: Acumular puntaje para TODOS los ítems similares ***
                # Ya no verificamos si 'similar_item_idx' está en el historial del usuario
                candidate_items[similar_item_idx] += similarity_score

        # Si después de iterar no hay candidatos, devolver vacío
        if not candidate_items:
             print(f"Advertencia: No se encontraron ítems candidatos basados en similitud para el usuario {target_cliente_id}.")
             return recommendations

        # --- Rankear Candidatos ---
        # Convertir el diccionario de candidatos (índice -> puntaje) a una lista de diccionarios
        ranked_candidates = []
        for item_idx, score in candidate_items.items():
             item_id = idx_to_item.get(item_idx) # Convertir índice interno de nuevo a ID de producto
             if item_id: # Asegurarse de que el mapeo inverso funcione
                 # Asegurar que el puntaje sea un float estándar para evitar problemas de serialización JSON
                 ranked_candidates.append({'producto': item_id, 'recommendation_score': float(score)})

        # Si no se pudo mapear ningún índice, devolver vacío
        if not ranked_candidates: return recommendations

        # Crear DataFrame, ordenar por puntaje descendente y tomar los N mejores
        recommendations = pd.DataFrame(ranked_candidates)
        recommendations = recommendations.sort_values('recommendation_score', ascending=False).head(N)

    except Exception as e:
         print(f"Error generando candidatos para el usuario {target_cliente_id}: {e}")
         traceback.print_exc()
         return pd.DataFrame() # Devolver DataFrame vacío en caso de error

    return recommendations

# --- Aplicación Flask ---
app = Flask(__name__)

# --- Endpoint de Recomendación ---
@app.route('/recommend/user/<int:cliente_id>', methods=['GET'])
def recommend_for_user(cliente_id):
    """Endpoint API para recomendaciones basadas en ítems para un cliente específico."""
    print(f"\nRecibida solicitud de recomendación Ítem-Based para Cliente ID: {cliente_id}")

    # Verificar si la configuración inicial de datos fue exitosa
    if not data_ready:
        print("Error: Los datos del motor de recomendación no están listos.")
        return jsonify({"error": "El motor de recomendación no está listo o falló la inicialización."}), 503 # Código 503: Servicio No Disponible

    # --- Obtener Parámetros Opcionales de la Query String ---
    # Obtener N (número de recomendaciones)
    try:
        n_recommendations = request.args.get('N', default=10, type=int) # Default 10 si no se especifica
        if n_recommendations <= 0: raise ValueError("N debe ser positivo")
    except (ValueError, TypeError):
        print(f"Parámetro 'N' inválido: {request.args.get('N')}")
        return jsonify({"error": "Parámetro 'N' inválido. Debe ser un entero positivo."}), 400 # Código 400: Solicitud Incorrecta

    # Obtener k (número de ítems similares a considerar)
    try:
        k_items = request.args.get('k', default=30, type=int) # Default 30 si no se especifica
        if k_items <= 0: raise ValueError("k debe ser positivo")
    except (ValueError, TypeError):
        print(f"Parámetro 'k' inválido: {request.args.get('k')}")
        return jsonify({"error": "Parámetro 'k' inválido. Debe ser un entero positivo."}), 400

    # --- Validar Cliente ID ---
    # Verificar si el cliente existe en nuestro mapeo (creado durante el setup)
    if cliente_id not in user_to_idx:
        print(f"Cliente ID '{cliente_id}' no encontrado.")
        return jsonify({"error": f"Cliente ID '{cliente_id}' no encontrado."}), 404 # Código 404: No Encontrado

    # --- Generar Recomendaciones ---
    start_rec_time = time.time()
    try:
        # Llamar a la función de recomendación específica (Ítem-Based)
        recommendations_df = get_item_based_recommendations_allow_repurchase(
            target_cliente_id=cliente_id,
            N=n_recommendations,
            k_similar_items=k_items
        )
        end_rec_time = time.time()

        # --- Preparar Respuesta ---
        # Si no se generaron recomendaciones (DataFrame vacío o None)
        if recommendations_df is None or recommendations_df.empty:
            print(f"No se generaron recomendaciones Ítem-Based para {cliente_id}.")
            response_data = {"cliente_id": cliente_id, "recommendations": []} # Devolver lista vacía
        else:
            # Convertir el DataFrame a una lista de diccionarios para la respuesta JSON
            result = recommendations_df.to_dict('records')
            print(f"Generadas {len(result)} recomendaciones Ítem-Based para {cliente_id} en {end_rec_time - start_rec_time:.4f}s.")
            response_data = {"cliente_id": cliente_id, "recommendations": result}

        # Devolver la respuesta JSON con código 200 OK
        return jsonify(response_data)

    except Exception as e:
        # Manejar errores inesperados durante la generación de recomendaciones
        print(f"ERROR durante la generación de recomendación Ítem-Based para {cliente_id}: {e}")
        traceback.print_exc()
        # Devolver un error genérico al cliente
        return jsonify({"error": "Ocurrió un error interno al generar las recomendaciones."}), 500 # Código 500: Error Interno del Servidor


# --- Endpoint de Verificación de Salud (Health Check) ---
@app.route('/health_item_based', methods=['GET'])
def health_check_item_based():
    """Endpoint básico para verificar el estado del servidor Ítem-Based."""
    status = "ok"
    details = {"engine_type": "item-based"}
    http_status = 200 # Asumir OK por defecto

    # Verificar si la configuración inicial fue exitosa
    if not data_ready:
        status = "error"
        details["data_status"] = "La inicialización falló o está incompleta"
        http_status = 503 # Servicio No Disponible si la data no está lista
    else:
        # Proveer detalles si la data está lista
        details["data_status"] = "Listo"
        details["users_mapped"] = n_users
        details["items_mapped"] = n_items
        # Mostrar forma de las matrices si existen (convertir a lista para JSON)
        details["user_item_matrix_shape"] = list(user_item_matrix_csr.shape) if user_item_matrix_csr is not None else "No disponible"
        details["item_similarity_matrix_shape"] = list(item_similarity_matrix.shape) if item_similarity_matrix is not None else "No disponible"

    # Devolver estado y detalles en formato JSON
    return jsonify({"status": status, "details": details}), http_status

# --- Bloque Principal de Ejecución ---
if __name__ == '__main__':
    # --- Llamar a la función de configuración de datos DIRECTAMENTE AL INICIO ---
    # Esto poblará las variables globales necesarias para la función de recomendación
    setup_recommendation_data()
    # La función imprimirá errores y pondrá data_ready=False si falla el setup.

    # --- Iniciar el Servidor Flask ---
    if data_ready:
        print(f"Iniciando servidor Flask Ítem-Based en http://0.0.0.0:{FLASK_PORT}...")
    else:
        # Advertir si el setup falló, el servidor iniciará pero las recomendaciones no funcionarán
        print(f"ADVERTENCIA: La configuración de datos falló. El servidor iniciará pero las recomendaciones podrían no funcionar.")
        print(f"Iniciando servidor Flask Ítem-Based en http://0.0.0.0:{FLASK_PORT}...")

    # Ejecutar la aplicación Flask
    # host='0.0.0.0' lo hace accesible desde fuera del localhost (útil para Docker/redes)
    # debug=False es recomendado para producción/estabilidad
    # use_reloader=False evita que el código de setup se ejecute dos veces en modo debug (aunque debug=False aquí)
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, use_reloader=False)