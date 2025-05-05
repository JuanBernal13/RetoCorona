from flask import Flask, request, jsonify
import pandas as pd
from collections import defaultdict
from flask_cors import CORS
import os
import numpy as np

app = Flask(__name__)
CORS(app)

# Nombre del archivo TXT
DATA_FILE = 'Datos/base_3_transaccional_b2b.txt'

# --- Variables globales para almacenar datos pre-calculados y el DataFrame ---
df_transactions = pd.DataFrame()
product_list = []
weighted_co_occurrence_matrix = {}
category_subcategory_map = defaultdict(set)
product_strategic_alignment = {}
product_average_value = {}
product_category_info = {}


def preprocess_data(df):
    """Pre-calcula datos útiles, incluyendo co-ocurrencia ponderada, alineación, valor y información de categoría."""
    global product_list, weighted_co_occurrence_matrix, category_subcategory_map, product_strategic_alignment, product_average_value, product_category_info

    if df.empty:
        print("DataFrame vacío, no se puede pre-procesar.")
        return

    print("Iniciando pre-procesamiento de datos...")

    # Limpiar y convertir columnas relevantes a string y numérico
    df['producto'] = df['producto'].astype(str).str.strip()
    df['id_b2b'] = df['id_b2b'].astype(str).str.strip()
    df['municipio'] = df['municipio'].astype(str).str.strip() # Aseguramos que municipio sea string
    df['zona'] = df['zona'].astype(str).str.strip() # Aseguramos que zona sea string
    df['categoria_b2b_macro'] = df['categoria_b2b_macro'].astype(str).str.strip() # Aseguramos string
    df['categoria_b2b'] = df['categoria_b2b'].astype(str).str.strip()
    df['subcategoria_b2b'] = df['subcategoria_b2b'].astype(str).str.strip()
    df['alineación con portafolio estratégico b2b'] = pd.to_numeric(df['alineación con portafolio estratégico b2b'], errors='coerce').fillna(0)
    df['valor_total'] = pd.to_numeric(df['valor_total'], errors='coerce').fillna(0)


    product_list = df['producto'].unique().tolist()

    # Pre-calcular co-ocurrencia ponderada por valor total
    if 'fecha_factura' in df.columns:
        df['fecha_factura'] = pd.to_datetime(df['fecha_factura'], errors='coerce')
        transactions = df.groupby(['id_b2b', 'fecha_factura'])
    else:
        transactions = df.groupby('id_b2b')

    print("Calculando matriz de co-ocurrencia ponderada por valor total...")
    for name, group in transactions:
        products_in_transaction = group['producto'].tolist()
        transaction_value = group['valor_total'].sum()

        for i in range(len(products_in_transaction)):
            for j in range(i + 1, len(products_in_transaction)):
                p1 = products_in_transaction[i]
                p2 = products_in_transaction[j]
                if p1 != p2:
                    if p1 not in weighted_co_occurrence_matrix:
                        weighted_co_occurrence_matrix[p1] = defaultdict(float)
                    if p2 not in weighted_co_occurrence_matrix:
                        weighted_co_occurrence_matrix[p2] = defaultdict(float)
                    weighted_co_occurrence_matrix[p1][p2] += transaction_value
                    weighted_co_occurrence_matrix[p2][p1] += transaction_value


    print("Pre-calculando mapa de categoría/subcategoría...")
    for index, row in df.iterrows():
        category_subcategory_map[(row['categoria_b2b'], row['subcategoria_b2b'])].add(row['producto'])

    print("Calculando alineación estratégica promedio por producto...")
    product_strategic_alignment = df.groupby('producto')['alineación con portafolio estratégico b2b'].mean().to_dict()

    print("Calculando valor total promedio por producto...")
    product_average_value = df.groupby('producto')['valor_total'].mean().to_dict()

    print("Construyendo mapa de información de categoría por producto...")
    product_info_df = df.drop_duplicates(subset=['producto'])
    product_category_info = {row['producto']: (row['categoria_b2b'], row['subcategoria_b2b'])
                             for index, row in product_info_df.iterrows()}


    print("Pre-procesamiento completo.")


# Cargar los datos al iniciar la aplicación y pre-procesar
try:
    df_transactions = pd.read_csv(DATA_FILE, sep='\t', low_memory=False)
    print(f"Datos de transacciones cargados exitosamente desde {DATA_FILE}.")
    preprocess_data(df_transactions.copy())
except FileNotFoundError:
    print(f"Error: {DATA_FILE} no encontrado. Asegúrate de que el archivo esté en la misma carpeta.")
    df_transactions = pd.DataFrame()
except Exception as e:
    print(f"Error al cargar o pre-procesar los datos desde {DATA_FILE}: {e}")
    df_transactions = pd.DataFrame()


# Función para calcular el puntaje de similitud de categoría/subcategoría
def get_category_similarity_score(product1_info, product2_info):
    """Calcula un puntaje de similitud basado en categoría y subcategoría."""
    if not product1_info or not product2_info:
        return 0

    cat1, subcat1 = product1_info
    cat2, subcat2 = product2_info

    if cat1 == cat2 and subcat1 == subcat2:
        return 2
    elif cat1 == cat2:
        return 1
    else:
        return 0


# La función de recomendación
def get_recommendations(product_name, num_recommendations=5):
    """
    Genera recomendaciones de productos usando datos pre-calculados, ponderando por co-ocurrencia, alineación estratégica y similitud de categoría.
    Retorna una lista vacía si el producto no se encuentra.
    """
    product_name_str = str(product_name).strip()

    if product_name_str not in product_list:
         return []

    input_product_category_info = product_category_info.get(product_name_str)

    initial_recommendations = []
    if product_name_str in weighted_co_occurrence_matrix:
        co_occurring_products = weighted_co_occurrence_matrix[product_name_str]
        sorted_co_occurring = sorted(co_occurring_products.items(), key=lambda item: item[1], reverse=True)
        initial_recommendations = [product for product, weight in sorted_co_occurring if product != product_name_str]
    else:
         initial_recommendations = []

    if len(initial_recommendations) < num_recommendations and not df_transactions.empty and input_product_category_info:
        product_category, product_subcategory = input_product_category_info
        similar_category_products_set = category_subcategory_map.get((product_category, product_subcategory), set())

        for prod in similar_category_products_set:
            if prod != product_name_str and prod not in initial_recommendations:
                initial_recommendations.append(prod)
            if len(initial_recommendations) >= num_recommendations:
                break

    weighted_recommendations = []
    for rec_prod in initial_recommendations:
        alignment_score = product_strategic_alignment.get(rec_prod, 0)
        average_value_score = product_average_value.get(rec_prod, 0)
        co_occurrence_weight = weighted_co_occurrence_matrix.get(product_name_str, {}).get(rec_prod, 0)

        recommended_product_category_info = product_category_info.get(rec_prod)
        category_similarity_score = get_category_similarity_score(input_product_category_info, recommended_product_category_info)

        weighted_recommendations.append((category_similarity_score, alignment_score, average_value_score, co_occurrence_weight, rec_prod))

    sorted_weighted_recommendations = sorted(weighted_recommendations, key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)

    recommendations = [prod for cat_sim, alignment, avg_value, weight, prod in sorted_weighted_recommendations]

    return recommendations[:num_recommendations]


@app.route('/recommend', methods=['POST'])
def recommend_products_route():
    data = request.get_json()
    product_name = data.get('product_name')

    if not product_name:
        return jsonify({"error": "Se requiere el nombre del producto"}), 400

    recommendations = get_recommendations(str(product_name))

    # --- Construir la respuesta con los detalles especificados ---
    recommended_products_details = []
    if recommendations and not df_transactions.empty:
        # Filtramos el DataFrame para obtener solo las filas de los productos recomendados
        
        recommended_products_df = df_transactions[df_transactions['producto'].isin(recommendations)].drop_duplicates(subset=['producto'])

        # Columnas para enviar al frontend
        columns_to_send = [
            'municipio',
            'zona',
            'categoria_b2b_macro',
            'categoria_b2b',
            'subcategoria_b2b',
            'producto',
            'valor_total',
            'alineación con portafolio estratégico b2b'
        ]

        # Aseguramos el orden de las recomendaciones
        product_details_map = {row['producto']: row.to_dict() for index, row in recommended_products_df.iterrows()}

        for rec_prod_name in recommendations:
             if rec_prod_name in product_details_map:
                 product_details = product_details_map[rec_prod_name]
                 recommended_product_data = {col: product_details.get(col) for col in columns_to_send}
                 recommended_products_details.append(recommended_product_data)


    # Si no hay recomendaciones, la lista estará vacía
    return jsonify(recommended_products_details)


if __name__ == '__main__':
    app.run(debug=True)