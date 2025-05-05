# server/app_b2c.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time
import os
from flask import Flask, request, jsonify
import traceback
from flask_cors import CORS

# --- Configuration ---
DIRECCION_DATOS = 'Datos'
# Assuming B2C data is in the same files for this example based on your prompt
DIRECCION_B2C_TRANSACCIONES = os.path.join(DIRECCION_DATOS, 'transacciones_con_features.csv')
DIRECCION_B2C_COTIZACIONES = os.path.join(DIRECCION_DATOS, 'cotizaciones_con_features.csv')
FLASK_PORT_B2C = 5001 # B2C will run on 5001

# --- Global Variables for B2C Pre-computed Data ---
product_features_unified_b2c = None
product_to_idx_b2c = None
idx_to_product_b2c = None
n_products_b2c = 0
content_similarity_matrix_b2c = None
preprocess_ok_b2c = False
similarity_ok_b2c = False


# --- Helper Functions (Adapted for B2C Naming and Features) ---

def _aggregate_features_b2c(transacciones, cotizaciones):
    """Aggregates features for B2C products."""
    print("B2C Step 1: Aggregating Features...")
    # Select features more relevant for B2C content-based filtering
    # You can adjust this list based on what features in your data are useful for B2C
    relevant_cols_transacciones_b2c = ['producto'] + [
        'categoria_macro', 'categoria', 'subcategoria', 'color',
        'precio_promedio_venta', # Consumer price sensitivity
        # Maybe popularity metrics if they reflect consumer trends
        'popularidad_valor_prod_en_cat',
        'popularidad_unidad_prod_en_cat',
        'popularidad_valor_prod_global',
        'popularidad_unidad_prod_global'
    ]
    relevant_cols_cotizaciones_b2c = ['producto'] + [
        'categoria_macro', 'categoria', 'precio', # Using price from cotizaciones
    ]


    if 'producto' not in transacciones.columns or 'producto' not in cotizaciones.columns:
        raise ValueError("ERROR (B2C): La columna 'producto' no se encuentra en uno o ambos DataFrames.")

    available_cols_trans = [col for col in relevant_cols_transacciones_b2c if col in transacciones.columns]
    available_cols_cot = [col for col in relevant_cols_cotizaciones_b2c if col in cotizaciones.columns]

    trans_agg_dict = {
        'categoria_macro': 'first', 'categoria': 'first', 'subcategoria': 'first',
        'color': lambda x: x.mode()[0] if not x.mode().empty else 'Desconocido',
        'precio_promedio_venta': 'mean',
        'popularidad_valor_prod_en_cat': 'mean',
        'popularidad_unidad_prod_en_cat': 'mean',
        'popularidad_valor_prod_global': 'mean',
        'popularidad_unidad_prod_global': 'mean'
    }
    valid_trans_agg_dict = {k: v for k, v in trans_agg_dict.items() if k in available_cols_trans}
    product_features_trans = transacciones.groupby('producto').agg(valid_trans_agg_dict)

    cot_agg_dict = {
        'categoria_macro': 'first', 'categoria': 'first',
        'precio': 'mean',
    }
    cols_for_cot_agg = [col for col in available_cols_cot if col != 'producto']
    if 'precio' not in cols_for_cot_agg and 'precio' in cotizaciones.columns: cols_for_cot_agg.append('precio')

    valid_cot_agg_dict = {k: v for k, v in cot_agg_dict.items() if k in cols_for_cot_agg or k in available_cols_cot}
    valid_cot_agg_dict = {k: v for k, v in valid_cot_agg_dict.items() if k in cotizaciones.columns or k in available_cols_cot}


    product_features_cot = cotizaciones.groupby('producto').agg(valid_cot_agg_dict)
    product_features_cot = product_features_cot.rename(columns={'precio': 'precio_promedio_cot'})

    print("B2C Step 1: Aggregation Done.")
    return product_features_trans, product_features_cot


def _unify_and_index_b2c(product_features_trans, product_features_cot):
    """Unifies features and creates index for B2C products."""
    print("B2C Step 2: Unifying Features and Creating Index...")
    # B2C might not need to drop categories if they are primary features
    product_features_unified = product_features_trans.join(product_features_cot, how='outer', lsuffix='_trans', rsuffix='_cot')

    # Handle potential duplicate columns from join, prioritize one if needed
    # For simplicity, let's handle specific duplicates like 'categoria_macro' by using one
    for col in ['categoria_macro', 'categoria']:
        if f'{col}_trans' in product_features_unified.columns and f'{col}_cot' in product_features_unified.columns:
            # Use the one from transactions if available, otherwise from cotizaciones
            product_features_unified[col] = product_features_unified[f'{col}_trans'].fillna(product_features_unified[f'{col}_cot'])
            product_features_unified = product_features_unified.drop(columns=[f'{col}_trans', f'{col}_cot'])
        elif f'{col}_trans' in product_features_unified.columns:
             product_features_unified[col] = product_features_unified[f'{col}_trans']
             product_features_unified = product_features_unified.drop(columns=f'{col}_trans')
        elif f'{col}_cot' in product_features_unified.columns:
             product_features_unified[col] = product_features_unified[f'{col}_cot']
             product_features_unified = product_features_unified.drop(columns=f'{col}_cot')


    all_unique_products = product_features_unified.index.unique().tolist()
    product_to_idx = {product: i for i, product in enumerate(all_unique_products)}
    idx_to_product = {i: product for product, i in product_to_idx.items()}
    n_products = len(all_unique_products)
    product_features_unified = product_features_unified.reindex(all_unique_products)
    print(f"B2C Step 2: Unified {n_products} products.")
    return product_features_unified, product_to_idx, idx_to_product, n_products


def _impute_missing_b2c(product_features_unified):
    """Imputes missing values for B2C product features."""
    print("B2C Step 3: Imputing Missing Values...")
    if product_features_unified is None or product_features_unified.empty:
        print("Warning (B2C): product_features_unified is empty, skipping imputation.")
        return product_features_unified, [], []

    final_num_features = product_features_unified.select_dtypes(include=np.number).columns.tolist()
    final_cat_features = product_features_unified.select_dtypes(include=['object', 'category']).columns.tolist()

    imputation_count = 0
    for col in final_num_features:
        if product_features_unified[col].isnull().any():
            imputation_count += product_features_unified[col].isnull().sum()
            median_val = product_features_unified[col].median()
            fill_val = median_val if pd.notna(median_val) else 0
            product_features_unified[col] = product_features_unified[col].fillna(fill_val)

    for col in final_cat_features:
        if product_features_unified[col].isnull().any():
            imputation_count += product_features_unified[col].isnull().sum()
            mode_val = product_features_unified[col].mode()
            fill_value = mode_val[0] if not mode_val.empty else 'Desconocido'
            product_features_unified[col] = product_features_unified[col].fillna(fill_value)

    print(f"B2C Step 3: Imputed {imputation_count} NaN values.")
    nans_remaining = product_features_unified.isnull().sum().sum()
    if nans_remaining > 0:
         print(f"WARNING (B2C): Still {nans_remaining} NaNs remaining after imputation!")
    return product_features_unified, final_num_features, final_cat_features


def _preprocess_features_b2c(product_features_unified, final_num_features, final_cat_features):
    """Scales and encodes B2C product features."""
    global preprocess_ok_b2c
    print("B2C Step 4: Preprocessing Features (Scaling/Encoding)...")
    if product_features_unified is None or product_features_unified.empty:
        print("Warning (B2C): product_features_unified is empty, skipping preprocessing.")
        preprocess_ok_b2c = False
        return None

    if not final_num_features and not final_cat_features:
         print("Warning (B2C): No numerical or categorical features found for preprocessing.")
         preprocess_ok_b2c = False
         return None

    start_time_preprocess = time.time()
    transformers = []
    if final_num_features:
        transformers.append(('num', MinMaxScaler(), final_num_features))
    if final_cat_features:
         # B2C might have different category features than B2B
         transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), final_cat_features))

    if not transformers:
        print("Warning (B2C): No features to preprocess.")
        preprocess_ok_b2c = False
        return None

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    try:
        feature_matrix_sparse = preprocessor.fit_transform(product_features_unified)
        end_time_preprocess = time.time()
        print(f"B2C Step 4: Preprocessing done in {end_time_preprocess - start_time_preprocess:.2f}s. Matrix shape: {feature_matrix_sparse.shape}")
        preprocess_ok_b2c = True
        return feature_matrix_sparse
    except Exception as e:
        print(f"ERROR (B2C) during preprocessing: {e}")
        traceback.print_exc()
        preprocess_ok_b2c = False
        return None


def _calculate_content_similarity_b2c(feature_matrix_sparse):
    """Calculates content similarity matrix for B2C products."""
    global similarity_ok_b2c
    print("B2C Step 5: Calculating Content Similarity...")
    start_time_similarity = time.time()
    content_similarity_matrix = None
    try:
        if feature_matrix_sparse is not None and feature_matrix_sparse.shape[0] > 0 and feature_matrix_sparse.shape[1] > 0:
             if isinstance(feature_matrix_sparse, (lil_matrix)):
                 feature_matrix_sparse = feature_matrix_sparse.tocsr()

             content_similarity_matrix = cosine_similarity(feature_matrix_sparse)
             end_time_similarity = time.time()
             print(f"B2C Step 5: Cosine similarity done in {end_time_similarity - start_time_similarity:.2f}s. Matrix shape: {content_similarity_matrix.shape}")
             similarity_ok_b2c = True
        else:
             print("ERROR (B2C): Feature matrix is empty or invalid for similarity calculation.")
             similarity_ok_b2c = False
    except Exception as e:
        print(f"ERROR (B2C) during similarity calculation: {e}")
        traceback.print_exc()
        similarity_ok_b2c = False
    return content_similarity_matrix

# --- B2C Recommendation Function (Content-Based) ---

def get_recommendations_content_b2c(input_product_name, N=10):
    """Generates B2C content-based recommendations."""
    print(f"Generating B2C recommendations for: {input_product_name}")
    recommendations_list = []

    if product_to_idx_b2c is None:
        print("B2C product map not initialized.")
        return []

    if input_product_name not in product_to_idx_b2c:
        print(f"B2C Product '{input_product_name}' not found in product map.")
        # Decide on fallback: popular products, random, or empty
        # For now, return empty list
        return []

    if not similarity_ok_b2c or content_similarity_matrix_b2c is None:
        print("B2C Content similarity matrix not ready.")
        return []

    try:
        idx = product_to_idx_b2c[input_product_name]
        if idx >= content_similarity_matrix_b2c.shape[0]:
            print(f"B2C Index {idx} out of bounds for similarity matrix shape {content_similarity_matrix_b2c.shape}")
            return []

        # Get similarity scores for the input product
        sim_vector = np.asarray(content_similarity_matrix_b2c[idx]).flatten()
        sim_scores = list(enumerate(sim_vector))

        # Sort products by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the top N similar products (excluding the input product itself)
        print(f"Found {len(sim_scores)} similar products for {input_product_name} (B2C).")
        for i, score in sim_scores:
            if i == idx: # Skip the input product itself
                continue
            if len(recommendations_list) >= N: # Stop after getting N recommendations
                break
            prod_name = idx_to_product_b2c.get(i)
            if prod_name: # Ensure product name exists
                # For B2C, we might want to include more details like brand, price, image
                # This requires fetching data from the original aggregated features DataFrame
                product_details = {}
                if product_features_unified_b2c is not None and prod_name in product_features_unified_b2c.index:
                    # Extract relevant B2C fields from the unified features
                    # Adapt these column names to what you want to show in the B2C card
                    details_row = product_features_unified_b2c.loc[prod_name]
                    product_details = {
                        'nombre': prod_name, # Or a display name if available
                        'score': float(score), # Include score for debugging/ranking
                        'categoria_macro': details_row.get('categoria_macro', 'N/A'),
                        'categoria': details_row.get('categoria', 'N/A'),
                        'subcategoria': details_row.get('subcategoria', 'N/A'),
                        'color': details_row.get('color', 'N/A'),
                        'precio': details_row.get('precio_promedio_venta', details_row.get('precio_promedio_cot', 'N/A')), # Use either sale or quote price
                         # Add image URL if available in your features or another lookup
                        'imagen': 'https://via.placeholder.com/150' # Placeholder for now
                    }
                     # Ensure numerical fields are formatted as strings if needed
                    if isinstance(product_details['precio'], (int, float)):
                         product_details['precio'] = f"{product_details['precio']:.2f}" # Example formatting

                # Only add if we got details or at least the product name
                if product_details:
                     recommendations_list.append(product_details)
                elif prod_name: # Fallback: just add the product name
                     recommendations_list.append({'nombre': prod_name, 'score': float(score)})


        print(f"Generated {len(recommendations_list)} B2C recommendations for {input_product_name}.")
        return recommendations_list

    except Exception as e:
        print(f"ERROR during B2C recommendation generation for {input_product_name}: {e}")
        traceback.print_exc()
        return []


# --- Flask Application ---
app_b2c = Flask(__name__)
CORS(app_b2c) # Enable CORS

def load_and_prepare_data_b2c():
    """Loads data and computes matrices for B2C ONCE at startup."""
    global product_features_unified_b2c, product_to_idx_b2c, idx_to_product_b2c, n_products_b2c
    global content_similarity_matrix_b2c, preprocess_ok_b2c, similarity_ok_b2c

    print("--- Starting B2C Data Loading and Preprocessing ---")
    start_time = time.time()

    try:
        print("Loading B2C CSV files...")
        # Assuming B2C data comes from the same files for this example
        if not os.path.exists(DIRECCION_B2C_TRANSACCIONES):
            print(f"WARNING: Transaction file not found: {DIRECCION_B2C_TRANSACCIONES}. B2C data incomplete.")
            transacciones = pd.DataFrame(columns=['pedido', 'producto'])
        else:
             transacciones = pd.read_csv(DIRECCION_B2C_TRANSACCIONES, encoding='utf-8')
             print(f"Loaded transacciones: {transacciones.shape}")

        if not os.path.exists(DIRECCION_B2C_COTIZACIONES):
             print(f"WARNING: Quotation file not found: {DIRECCION_B2C_COTIZACIONES}. B2C data incomplete.")
             cotizaciones = pd.DataFrame(columns=['cotizacion', 'producto'])
        else:
             cotizaciones = pd.read_csv(DIRECCION_B2C_COTIZACIONES, encoding='utf-8')
             print(f"Loaded cotizaciones: {cotizaciones.shape}")

        if not transacciones.empty or not cotizaciones.empty:
             product_features_trans, product_features_cot = _aggregate_features_b2c(transacciones, cotizaciones)
             product_features_unified_b2c, product_to_idx_b2c, idx_to_product_b2c, n_products_b2c = _unify_and_index_b2c(product_features_trans, product_features_cot)
             print(f"Total unique B2C products identified: {n_products_b2c}")

             product_features_unified_b2c, num_feat, cat_feat = _impute_missing_b2c(product_features_unified_b2c)
             feature_matrix_sparse = _preprocess_features_b2c(product_features_unified_b2c, num_feat, cat_feat)

             if preprocess_ok_b2c and feature_matrix_sparse is not None:
                 content_similarity_matrix_b2c = _calculate_content_similarity_b2c(feature_matrix_sparse)
             else:
                 print("Skipping B2C content similarity due to preprocessing issues.")
                 similarity_ok_b2c = False

        else:
             print("WARNING: No B2C transaction or quotation data found. B2C recommendations will be unavailable.")
             product_features_unified_b2c = None
             product_to_idx_b2c = {}
             idx_to_product_b2c = {}
             n_products_b2c = 0
             content_similarity_matrix_b2c = None
             preprocess_ok_b2c = False
             similarity_ok_b2c = False


        end_time = time.time()
        print(f"--- B2C Data Loading and Preprocessing COMPLETE in {end_time - start_time:.2f} seconds ---")
        print(f"B2C Status Flags: preprocess={preprocess_ok_b2c}, similarity={similarity_ok_b2c}")
        if not similarity_ok_b2c:
             print("WARNING: B2C content similarity matrix not built. B2C API may not return results.")


    except Exception as e:
        print(f"CRITICAL ERROR during initial B2C data preparation: {e}")
        traceback.print_exc()
        print("B2C Server starting with potential data loading errors. Recommendation endpoints may fail.")
        product_features_unified_b2c = None
        product_to_idx_b2c = {}
        idx_to_product_b2c = {}
        n_products_b2c = 0
        content_similarity_matrix_b2c = None
        preprocess_ok_b2c = False
        similarity_ok_b2c = False


@app_b2c.route('/recommend', methods=['POST'])
def recommend_b2c():
    """API endpoint to get B2C recommendations."""
    data = request.get_json()
    if not data or 'product_name' not in data:
        print("Invalid B2C request: Missing product_name in body.")
        return jsonify({"error": "Invalid request. Please provide 'product_name'."}), 400

    product_name = data['product_name']
    # Assuming user_type 'person' for this endpoint

    print(f"\nReceived B2C recommendation request for product: '{product_name}'")

    try:
        n_recommendations = request.args.get('N', default=10, type=int)
        if n_recommendations <= 0:
            raise ValueError("N must be a positive integer.")
    except (ValueError, TypeError):
        print(f"Invalid 'N' parameter received: {request.args.get('N')}")
        return jsonify({"error": "Invalid parameter 'N'. Must be a positive integer."}), 400

    start_rec_time = time.time()
    recommendations_list = []

    try:
        if product_to_idx_b2c is None or product_name not in product_to_idx_b2c:
            print(f"B2C Product ID '{product_name}' not found in product map.")
            recommendations_list = [] # Return empty list if product not found
        elif not similarity_ok_b2c:
             print("Warning: B2C recommendation components are not ready. Cannot generate B2C recommendations.")
             recommendations_list = [] # Return empty list
        else:
            recommendations_list = get_recommendations_content_b2c(
                input_product_name=product_name,
                N=n_recommendations
            )
            # The get_recommendations_content_b2c function already returns the desired structure

        end_rec_time = time.time()
        print(f"Generated {len(recommendations_list)} B2C recommendations for {product_name} in {end_rec_time - start_rec_time:.4f}s.")

        # Return the result as a list of dictionaries within a key
        # Structure matches frontend expectation { recommendations: [...] }
        # Frontend expects 'nombre', 'marca', 'precio', 'imagen' for B2C
        # Let's ensure the returned list of dicts has these keys
        # The get_recommendations_content_b2c function is adapted to build this structure
        return jsonify({"product_id": product_name, "recommendations": recommendations_list})

    except Exception as e:
        print(f"ERROR during B2C recommendation generation for {product_name}: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred while generating B2C recommendations."}), 500


@app_b2c.route('/health', methods=['GET'])
def health_check_b2c():
    """Basic B2C health check endpoint."""
    status = "ok"
    details = {}

    b2c_ready = product_to_idx_b2c is not None and product_to_idx_b2c and similarity_ok_b2c
    details["b2c_ready"] = b2c_ready
    details["b2c_status_flags"] = f"preprocess={preprocess_ok_b2c}, similarity={similarity_ok_b2c}"
    details["b2c_product_count"] = n_products_b2c

    if not b2c_ready:
        status = "error"
        details["overall_status_reason"] = "B2C recommendations are not ready."

    http_status = 200 if status == "ok" else 503

    return jsonify({"status": status, "details": details}), http_status


if __name__ == '__main__':
    print("--- Initializing B2C Recommendation Engine ---")
    load_and_prepare_data_b2c()
    print("--- B2C Initialization Potentially Complete (Check Status Flags Above) ---")

    print(f"Starting B2C Flask server on http://0.0.0.0:{FLASK_PORT_B2C}...")
    # use_reloader=False is important when running multiple Flask apps
    app_b2c.run(host='0.0.0.0', port=FLASK_PORT_B2C, debug=False, use_reloader=False)