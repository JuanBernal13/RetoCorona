import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import lil_matrix, csr_matrix
from collections import defaultdict
import time
import os
from flask import Flask, request, jsonify
import traceback # Import traceback for better error logging

# --- Configuration ---
DIRECCION_DATOS = 'Datos' # Directory containing the CSV files
DIRECCION_B2C_TRANSACCIONES = os.path.join(DIRECCION_DATOS, 'transacciones_con_features.csv')
DIRECCION_B2C_COTIZACIONES = os.path.join(DIRECCION_DATOS, 'cotizaciones_con_features.csv')
FLASK_PORT = 5001

# --- Global Variables for Pre-computed Data ---
# These will be populated once when the server starts
product_features_unified = None
product_to_idx = None
idx_to_product = None
n_products = 0
content_similarity_matrix = None
co_occurrence_matrix = None
co_quotation_matrix = None
similarity_ok = False
cf_buy_ok = False
cf_quote_ok = False
preprocess_ok = False

# --- Helper Functions (Copied/Adapted from Notebook) ---

# NOTE: Removed print statements from these functions for cleaner server logs
#       Error messages can be logged or returned in API responses.

# Cell 7: Feature Selection and Aggregation Logic
def _aggregate_features(transacciones, cotizaciones):
    print("Step 1: Aggregating Features...")
    relevant_cols_transacciones_user = ['producto'] + [
        'categoria_macro', 'categoria', 'subcategoria', 'color',
        'precio_promedio_venta', 'alineación con portafolio estratégico',
        'total_unidades_vendidas', 'valor_total_ventas', 'n_transacciones_producto',
        'n_clientes_producto', 'frecuencia_venta_prod', 'popularidad_valor_prod_en_cat',
        'popularidad_unidad_prod_en_cat', 'popularidad_valor_prod_global',
        'popularidad_unidad_prod_global'
    ]
    relevant_cols_cotizaciones_user = ['producto'] + [
        'categoria_macro', 'categoria', 'precio', 'valor', # Added precio/valor for aggregation
        'total_unidades_cotizadas', 'valor_total_cotizado',
        'n_cotizaciones_producto', 'producto_fue_comprado_por_cliente'
    ]

    if 'producto' not in transacciones.columns or 'producto' not in cotizaciones.columns:
        raise ValueError("ERROR: La columna 'producto' no se encuentra en uno o ambos DataFrames.")

    available_cols_trans = [col for col in relevant_cols_transacciones_user if col in transacciones.columns]
    available_cols_cot = [col for col in relevant_cols_cotizaciones_user if col in cotizaciones.columns]

    trans_agg_dict = {
        'categoria_macro': 'first', 'categoria': 'first', 'subcategoria': 'first',
        'color': lambda x: x.mode()[0] if not x.mode().empty else 'Desconocido',
        'precio_promedio_venta': 'mean', 'alineación con portafolio estratégico': 'mean',
        'total_unidades_vendidas': 'mean', 'valor_total_ventas': 'mean',
        'n_transacciones_producto': 'mean', 'n_clientes_producto': 'mean',
        'frecuencia_venta_prod': 'mean', 'popularidad_valor_prod_en_cat': 'mean',
        'popularidad_unidad_prod_en_cat': 'mean', 'popularidad_valor_prod_global': 'mean',
        'popularidad_unidad_prod_global': 'mean'
    }
    valid_trans_agg_dict = {k: v for k, v in trans_agg_dict.items() if k in available_cols_trans}
    product_features_trans = transacciones.groupby('producto').agg(valid_trans_agg_dict)

    cot_agg_dict = {
        'categoria_macro': 'first', 'categoria': 'first',
        'precio': 'mean', 'valor': 'mean',
        'total_unidades_cotizadas': 'mean', 'valor_total_cotizado': 'mean',
        'n_cotizaciones_producto': 'mean', 'producto_fue_comprado_por_cliente': 'max'
    }
    cols_for_cot_agg = [col for col in available_cols_cot if col != 'producto']
    if 'precio' not in cols_for_cot_agg and 'precio' in cotizaciones.columns: cols_for_cot_agg.append('precio')
    if 'valor' not in cols_for_cot_agg and 'valor' in cotizaciones.columns: cols_for_cot_agg.append('valor')

    valid_cot_agg_dict = {k: v for k, v in cot_agg_dict.items() if k in cols_for_cot_agg or k in available_cols_cot}
    valid_cot_agg_dict = {k: v for k, v in valid_cot_agg_dict.items() if k in cotizaciones.columns or k in available_cols_cot}

    product_features_cot = cotizaciones.groupby('producto').agg(valid_cot_agg_dict)
    product_features_cot = product_features_cot.rename(columns={'precio': 'precio_promedio_cot', 'valor': 'valor_promedio_cot'})

    print("Step 1: Aggregation Done.")
    return product_features_trans, product_features_cot

# Cell 8 & 9: Unify Features & Create Index
def _unify_and_index(product_features_trans, product_features_cot):
    print("Step 2: Unifying Features and Creating Index...")
    product_features_cot_1 = product_features_cot.drop(columns=['categoria_macro', 'categoria'], errors='ignore')
    product_features_unified = product_features_trans.join(product_features_cot_1, how='outer')

    all_unique_products = product_features_unified.index.unique().tolist()
    product_to_idx = {product: i for i, product in enumerate(all_unique_products)}
    idx_to_product = {i: product for product, i in product_to_idx.items()}
    n_products = len(all_unique_products)
    product_features_unified = product_features_unified.reindex(all_unique_products)
    print(f"Step 2: Unified {n_products} products.")
    return product_features_unified, product_to_idx, idx_to_product, n_products

# Cell 10: Impute Missing Values
def _impute_missing(product_features_unified):
    print("Step 3: Imputing Missing Values...")
    # Need to handle potential empty DataFrame after join/reindex
    if product_features_unified is None or product_features_unified.empty:
        print("Warning: product_features_unified is empty, skipping imputation.")
        return product_features_unified, [], []

    final_num_features = product_features_unified.select_dtypes(include=np.number).columns.tolist()
    final_cat_features = product_features_unified.select_dtypes(include=['object', 'category']).columns.tolist()

    imputation_count = 0
    for col in final_num_features:
        if product_features_unified[col].isnull().any():
            imputation_count += product_features_unified[col].isnull().sum() # Count actual NaNs imputed
            median_val = product_features_unified[col].median()
            fill_val = median_val if pd.notna(median_val) else 0
            product_features_unified[col] = product_features_unified[col].fillna(fill_val)

    for col in final_cat_features:
        if product_features_unified[col].isnull().any():
            imputation_count += product_features_unified[col].isnull().sum()
            mode_val = product_features_unified[col].mode()
            fill_value = mode_val[0] if not mode_val.empty else 'Desconocido'
            product_features_unified[col] = product_features_unified[col].fillna(fill_value)

    print(f"Step 3: Imputed {imputation_count} NaN values.")
    nans_remaining = product_features_unified.isnull().sum().sum()
    if nans_remaining > 0:
         print(f"WARNING: Still {nans_remaining} NaNs remaining after imputation!")
    return product_features_unified, final_num_features, final_cat_features

# Cell 12: Preprocessing (Scaling & Encoding)
def _preprocess_features(product_features_unified, final_num_features, final_cat_features):
    global preprocess_ok # Allow modifying global flag
    print("Step 4: Preprocessing Features (Scaling/Encoding)...")
    # Handle empty DataFrame case
    if product_features_unified is None or product_features_unified.empty:
        print("Warning: product_features_unified is empty, skipping preprocessing.")
        preprocess_ok = False
        return None
    # Handle case where feature lists might be empty if input was empty
    if not final_num_features and not final_cat_features:
         print("Warning: No numerical or categorical features found for preprocessing.")
         preprocess_ok = False # Or True depending on desired behavior? Let's say False.
         return None # Or maybe return an empty sparse matrix? For now None.

    start_time_preprocess = time.time()
    transformers = []
    if final_num_features:
        transformers.append(('num', MinMaxScaler(), final_num_features))
    if final_cat_features:
         transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), final_cat_features))

    # Only proceed if there's something to transform
    if not transformers:
        print("Warning: No features to preprocess.")
        preprocess_ok = False
        return None

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    try:
        feature_matrix_sparse = preprocessor.fit_transform(product_features_unified)
        end_time_preprocess = time.time()
        print(f"Step 4: Preprocessing done in {end_time_preprocess - start_time_preprocess:.2f}s. Matrix shape: {feature_matrix_sparse.shape}")
        preprocess_ok = True
        return feature_matrix_sparse
    except Exception as e:
        print(f"ERROR during preprocessing: {e}")
        traceback.print_exc() # Print full traceback for debugging
        preprocess_ok = False
        return None

# Cell 13: Content Similarity Calculation
def _calculate_content_similarity(feature_matrix_sparse):
    global similarity_ok # Allow modifying global flag
    print("Step 5: Calculating Content Similarity...")
    start_time_similarity = time.time()
    content_similarity_matrix = None
    try:
        if feature_matrix_sparse is not None and feature_matrix_sparse.shape[0] > 0 and feature_matrix_sparse.shape[1] > 0:
             # Ensure the matrix is CSR for cosine_similarity if it's sparse
             if isinstance(feature_matrix_sparse, (lil_matrix)):
                 feature_matrix_sparse = feature_matrix_sparse.tocsr()

             content_similarity_matrix = cosine_similarity(feature_matrix_sparse)
             end_time_similarity = time.time()
             print(f"Step 5: Cosine similarity done in {end_time_similarity - start_time_similarity:.2f}s. Matrix shape: {content_similarity_matrix.shape}")
             similarity_ok = True
        else:
             print("ERROR: Feature matrix is empty or invalid for similarity calculation.")
             similarity_ok = False
    except Exception as e:
        print(f"ERROR during similarity calculation: {e}")
        traceback.print_exc()
        similarity_ok = False
    return content_similarity_matrix

# Cell 16: Co-purchase Calculation
def _calculate_co_purchase(transacciones, product_to_idx, n_products):
    global cf_buy_ok # Allow modifying global flag
    print("Step 6: Calculating Co-Purchase Matrix...")
    start_time_cf_buy = time.time()
    co_occurrence_matrix = None
    try:
        # Add basic checks for inputs
        if transacciones is None or transacciones.empty or product_to_idx is None or n_products <= 0:
             print("Warning: Invalid inputs for co-purchase calculation.")
             cf_buy_ok = False
             return None

        if 'pedido' in transacciones.columns and 'producto' in transacciones.columns:
            # Filter transactions for known products *before* grouping
            known_products = set(product_to_idx.keys())
            transacciones_filtered = transacciones[transacciones['producto'].isin(known_products)]

            if transacciones_filtered.empty:
                print("Warning: No transactions found with known products for co-purchase.")
                cf_buy_ok = False
                # Return an empty matrix of the correct size
                return csr_matrix((n_products, n_products), dtype=np.int32)

            co_occurrence_matrix_lil = lil_matrix((n_products, n_products), dtype=np.int32)
            pedidos_grouped = transacciones_filtered.groupby('pedido')['producto'].unique()

            processed_pairs_buy = 0
            for products_in_pedido in pedidos_grouped:
                # Map products to indices
                indices_in_pedido = [product_to_idx[p] for p in products_in_pedido if p in product_to_idx] # Check again just in case

                # Only proceed if there's more than one product in the order
                if len(indices_in_pedido) > 1:
                    for i in range(len(indices_in_pedido)):
                        for j in range(i + 1, len(indices_in_pedido)):
                            idx1, idx2 = indices_in_pedido[i], indices_in_pedido[j]
                            co_occurrence_matrix_lil[idx1, idx2] += 1
                            co_occurrence_matrix_lil[idx2, idx1] += 1
                            processed_pairs_buy += 1 # Count pairs resulting in increment

            co_occurrence_matrix = co_occurrence_matrix_lil.tocsr()
            end_time_cf_buy = time.time()
            density = co_occurrence_matrix.nnz / (n_products * n_products) if n_products > 0 else 0
            print(f"Step 6: Co-purchase matrix ({co_occurrence_matrix.shape}) done in {end_time_cf_buy - start_time_cf_buy:.2f}s. Non-zero entries: {co_occurrence_matrix.nnz}, Pairs found: {processed_pairs_buy}, Density: {density:.6f}")
            cf_buy_ok = True
        else:
            print("Warning: Cannot calculate co-purchase. Missing 'pedido' or 'producto' columns.")
            cf_buy_ok = False
            # Return an empty matrix if columns are missing
            co_occurrence_matrix = csr_matrix((n_products, n_products), dtype=np.int32)

    except Exception as e:
        print(f"ERROR during co-purchase calculation: {e}")
        traceback.print_exc()
        cf_buy_ok = False
        # Ensure co_occurrence_matrix is None or empty on error
        co_occurrence_matrix = None # Or return csr_matrix((n_products, n_products), dtype=np.int32)

    return co_occurrence_matrix


# Cell 17: Co-quotation Calculation
def _calculate_co_quotation(cotizaciones, product_to_idx, n_products):
    global cf_quote_ok # Allow modifying global flag
    print("Step 7: Calculating Co-Quotation Matrix...")
    start_time_cf_quote = time.time()
    co_quotation_matrix = None
    try:
        # Add basic checks for inputs
        if cotizaciones is None or cotizaciones.empty or product_to_idx is None or n_products <= 0:
             print("Warning: Invalid inputs for co-quotation calculation.")
             cf_quote_ok = False
             return None

        if 'cotizacion' in cotizaciones.columns and 'producto' in cotizaciones.columns:
            # Filter for known products
            known_products = set(product_to_idx.keys())
            cotizaciones_filtered = cotizaciones[cotizaciones['producto'].isin(known_products)]

            if cotizaciones_filtered.empty:
                print("Warning: No quotations found with known products for co-quotation.")
                cf_quote_ok = False
                return csr_matrix((n_products, n_products), dtype=np.int32)

            co_quotation_matrix_lil = lil_matrix((n_products, n_products), dtype=np.int32)
            cotizaciones_grouped = cotizaciones_filtered.groupby('cotizacion')['producto'].unique()

            processed_pairs_quote = 0
            for products_in_quote in cotizaciones_grouped:
                indices_in_quote = [product_to_idx[p] for p in products_in_quote if p in product_to_idx]

                if len(indices_in_quote) > 1:
                    for i in range(len(indices_in_quote)):
                        for j in range(i + 1, len(indices_in_quote)):
                            idx1, idx2 = indices_in_quote[i], indices_in_quote[j]
                            co_quotation_matrix_lil[idx1, idx2] += 1
                            co_quotation_matrix_lil[idx2, idx1] += 1
                            processed_pairs_quote += 1

            co_quotation_matrix = co_quotation_matrix_lil.tocsr()
            end_time_cf_quote = time.time()
            density = co_quotation_matrix.nnz / (n_products * n_products) if n_products > 0 else 0
            print(f"Step 7: Co-quotation matrix ({co_quotation_matrix.shape}) done in {end_time_cf_quote - start_time_cf_quote:.2f}s. Non-zero entries: {co_quotation_matrix.nnz}, Pairs found: {processed_pairs_quote}, Density: {density:.6f}")
            cf_quote_ok = True
        else:
            print("Warning: Cannot calculate co-quotation. Missing 'cotizacion' or 'producto' columns.")
            cf_quote_ok = False
            co_quotation_matrix = csr_matrix((n_products, n_products), dtype=np.int32)

    except Exception as e:
        print(f"ERROR during co-quotation calculation: {e}")
        traceback.print_exc()
        cf_quote_ok = False
        co_quotation_matrix = None # Or return csr_matrix((n_products, n_products), dtype=np.int32)

    return co_quotation_matrix


# --- Recommendation Functions (Copied from Notebook) ---

# Cell 14: Content Recommendations (Helper for Hybrid)
def _get_content_recs_for_hybrid(idx, N):
    recs = []
    if similarity_ok and content_similarity_matrix is not None and idx < content_similarity_matrix.shape[0]:
        try:
            # Ensure content_similarity_matrix[idx] is treated as a 1D array
            sim_vector = np.asarray(content_similarity_matrix[idx]).flatten()
            sim_scores = list(enumerate(sim_vector))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            for i, score in sim_scores:
                if i == idx: continue
                prod_name = idx_to_product.get(i)
                if prod_name and len(recs) < N:
                    # Ensure score is a standard float
                    recs.append({'producto': prod_name, 'score': float(score)})
                elif len(recs) >= N:
                    break
        except Exception as e:
            print(f"Error in _get_content_recs_for_hybrid for index {idx}: {e}") # Log error
            traceback.print_exc()
    else:
        if not similarity_ok: print("Content similarity matrix not OK.")
        if content_similarity_matrix is None: print("Content similarity matrix is None.")
        if idx >= content_similarity_matrix.shape[0]: print(f"Index {idx} out of bounds for similarity matrix shape {content_similarity_matrix.shape}")

    return pd.DataFrame(recs) if recs else pd.DataFrame(columns=['producto', 'score'])


# Cell 18: Co-Purchase Recommendations (Helper for Hybrid)
def _get_co_purchase_recs_for_hybrid(idx, N):
    recs = []
    if cf_buy_ok and co_occurrence_matrix is not None and idx < co_occurrence_matrix.shape[0]:
        try:
            co_buys = co_occurrence_matrix[idx, :]
            co_buy_indices = co_buys.indices
            co_buy_values = co_buys.data
            if len(co_buy_indices) > 0:
                # Combine indices and values, sort by value (count)
                cobuy_pairs = sorted(zip(co_buy_indices, co_buy_values), key=lambda x: x[1], reverse=True)
                for i, count in cobuy_pairs:
                    if i == idx: continue # Exclude self
                    prod_name = idx_to_product.get(i)
                    if prod_name and len(recs) < N:
                        # Store count as score, ensure it's int
                        recs.append({'producto': prod_name, 'score': int(count)})
                    elif len(recs) >= N:
                        break # Stop when N recommendations are found
        except Exception as e:
            print(f"Error in _get_co_purchase_recs_for_hybrid for index {idx}: {e}") # Log error
            traceback.print_exc()
    # Return empty DataFrame if no recommendations or error
    return pd.DataFrame(recs) if recs else pd.DataFrame(columns=['producto', 'score'])


# Cell 18: Co-Quotation Recommendations (Helper for Hybrid)
def _get_co_quotation_recs_for_hybrid(idx, N):
    recs = []
    if cf_quote_ok and co_quotation_matrix is not None and idx < co_quotation_matrix.shape[0]:
        try:
            co_quotes = co_quotation_matrix[idx, :]
            co_quote_indices = co_quotes.indices
            co_quote_values = co_quotes.data
            if len(co_quote_indices) > 0:
                coquote_pairs = sorted(zip(co_quote_indices, co_quote_values), key=lambda x: x[1], reverse=True)
                for i, count in coquote_pairs:
                    if i == idx: continue
                    prod_name = idx_to_product.get(i)
                    if prod_name and len(recs) < N:
                         # Store count as score, ensure it's int
                        recs.append({'producto': prod_name, 'score': int(count)})
                    elif len(recs) >= N:
                        break
        except Exception as e:
            print(f"Error in _get_co_quotation_recs_for_hybrid for index {idx}: {e}") # Log error
            traceback.print_exc()
    return pd.DataFrame(recs) if recs else pd.DataFrame(columns=['producto', 'score'])

# Cell 13b (modified): Get Top-N Candidates Per Method
def get_top_n_candidates_per_method(input_product, N=10):
    results = {
        'content': pd.DataFrame(columns=['producto', 'score']),
        'co_purchase': pd.DataFrame(columns=['producto', 'score']),
        'co_quotation': pd.DataFrame(columns=['producto', 'score'])
    }
    if input_product not in product_to_idx:
        # Don't print here, let the main function handle it
        print(f"Warning: Product '{input_product}' not in product_to_idx map (get_top_n_candidates).")
        return results

    idx = product_to_idx[input_product]

    # Check if index is valid for matrices before calling helpers
    results['content'] = _get_content_recs_for_hybrid(idx, N)
    results['co_purchase'] = _get_co_purchase_recs_for_hybrid(idx, N)
    results['co_quotation'] = _get_co_quotation_recs_for_hybrid(idx, N)

    return results

# Cell 15: Hybrid Recommendation Function
def get_recommendations_hybrid_rerank(input_product, N=10,
                                      content_weight=0.3,
                                      cf_buy_weight=0.5,
                                      cf_quote_weight=0.2,
                                      k=2,
                                      fetch_top_M=50):
    hybrid_scores = defaultdict(float)

    # Check if product exists FIRST
    if product_to_idx is None or input_product not in product_to_idx:
        print(f"Product '{input_product}' not found in mapping.") # Log this
        return pd.DataFrame() # Return empty DataFrame

    # --- 1. Obtain Top-M candidates ---
    separate_candidates = get_top_n_candidates_per_method(input_product, N=fetch_top_M)

    # --- Adaptive Weighting (Example Logic) ---
    # Check if co_purchase list is not empty and has a 'score' column
    if separate_candidates['co_purchase'] is not None and not separate_candidates['co_purchase'].empty and 'score' in separate_candidates['co_purchase'].columns:
      # Check if the first score exists and is below threshold
      if len(separate_candidates['co_purchase']['score']) > 0 and separate_candidates['co_purchase']['score'].iloc[0] < 10:
          print(f"Low co-purchase count for {input_product}, adjusting weights.")
          content_weight=0.5
          cf_buy_weight=0.3
          cf_quote_weight=0.2 # Ensure weights still sum roughly to 1 if desired
      # else: # Optional: Log when weights are *not* adjusted
          # print(f"Sufficient co-purchase count for {input_product}, using default weights.")
    else:
        # Handle case where co_purchase recommendations are empty or lack score
        print(f"No co-purchase data found for {input_product} to evaluate for weight adjustment.")


    # --- 2. Calculate Hybrid Score ---
    all_candidates = set()
    total_weight_applied = 0 # Track if any method contributed
    for method, weight in [('content', content_weight),
                           ('co_purchase', cf_buy_weight),
                           ('co_quotation', cf_quote_weight)]:
        if weight <= 0: continue
        recs_df = separate_candidates.get(method)
        # Check if DataFrame is valid
        if recs_df is not None and not recs_df.empty and 'producto' in recs_df.columns:
            method_contributed = False # Track if this method added scores
            # Add candidates to the overall set
            all_candidates.update(recs_df['producto'].tolist())
            # Iterate through recommendations of this method
            for rank, product in enumerate(recs_df['producto'], 1): # Rank starts at 1
                 if product == input_product: continue # Skip self
                 # Calculate reciprocal rank score contribution
                 score_contribution = weight * (1.0 / (rank + k))
                 hybrid_scores[product] += score_contribution
                 method_contributed = True
            if method_contributed:
                total_weight_applied += weight # Or just set a flag that at least one method worked

    # --- 3. Generate Final Ranking ---
    if not hybrid_scores:
        # This case means no candidates were found by any method with weight > 0
        print(f"No candidates found by any active method for hybrid re-ranking of '{input_product}'.")
        return pd.DataFrame()

    # Convert the defaultdict of scores to a DataFrame
    final_recs_df = pd.DataFrame(hybrid_scores.items(), columns=['producto', 'hybrid_score'])

    # Ensure we don't include the input product itself (double-check)
    final_recs_df = final_recs_df[final_recs_df['producto'] != input_product]

    # Sort by the calculated hybrid score in descending order and take the top N
    final_recs_df = final_recs_df.sort_values('hybrid_score', ascending=False).head(N)

    return final_recs_df


# --- Flask Application ---
app = Flask(__name__)

# This function will run ONCE when the app starts
def load_and_prepare_data():
    """Loads data and computes matrices ONCE at startup."""
    global product_features_unified, product_to_idx, idx_to_product, n_products
    global content_similarity_matrix, co_occurrence_matrix, co_quotation_matrix
    global similarity_ok, cf_buy_ok, cf_quote_ok, preprocess_ok

    print("--- Starting Data Loading and Preprocessing ---")
    start_time = time.time()

    try:
        # Load Data
        print("Loading CSV files...")
        if not os.path.exists(DIRECCION_B2C_TRANSACCIONES):
            raise FileNotFoundError(f"Transaction file not found: {DIRECCION_B2C_TRANSACCIONES}")
        if not os.path.exists(DIRECCION_B2C_COTIZACIONES):
             raise FileNotFoundError(f"Quotation file not found: {DIRECCION_B2C_COTIZACIONES}")

        transacciones = pd.read_csv(DIRECCION_B2C_TRANSACCIONES, encoding='utf-8')
        cotizaciones = pd.read_csv(DIRECCION_B2C_COTIZACIONES, encoding='utf-8')
        print(f"Loaded transacciones: {transacciones.shape}, cotizaciones: {cotizaciones.shape}")

        # Step 1: Aggregate Features
        product_features_trans, product_features_cot = _aggregate_features(transacciones, cotizaciones)

        # Step 2: Unify Features & Create Index
        product_features_unified, product_to_idx, idx_to_product, n_products = _unify_and_index(product_features_trans, product_features_cot)
        print(f"Total unique products identified: {n_products}") # Add log for clarity

        # Step 3: Impute Missing Values
        product_features_unified, num_feat, cat_feat = _impute_missing(product_features_unified)

        # Step 4: Preprocess Features (Scale/Encode)
        feature_matrix_sparse = _preprocess_features(product_features_unified, num_feat, cat_feat)

        # Step 5: Calculate Content Similarity
        if preprocess_ok and feature_matrix_sparse is not None:
            content_similarity_matrix = _calculate_content_similarity(feature_matrix_sparse)
        else:
            print("Skipping content similarity due to preprocessing issues.")
            similarity_ok = False

        # Step 6: Calculate Co-Purchase
        co_occurrence_matrix = _calculate_co_purchase(transacciones, product_to_idx, n_products)

        # Step 7: Calculate Co-Quotation
        co_quotation_matrix = _calculate_co_quotation(cotizaciones, product_to_idx, n_products)

        end_time = time.time()
        print(f"--- Data Loading and Preprocessing COMPLETE in {end_time - start_time:.2f} seconds ---")
        print(f"Status Flags: preprocess={preprocess_ok}, similarity={similarity_ok}, cf_buy={cf_buy_ok}, cf_quote={cf_quote_ok}")
        if not (similarity_ok or cf_buy_ok or cf_quote_ok):
             print("WARNING: No recommendation components initialized successfully. API may not return results.")

    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: {e}")
        # Stop the server or handle gracefully
        raise SystemExit(f"CRITICAL ERROR: Cannot load data files. Server cannot start. Details: {e}") from e
    except Exception as e:
        print(f"CRITICAL ERROR during initial data preparation: {e}")
        traceback.print_exc()
        # Stop the server or handle gracefully depending on severity
        raise SystemExit(f"CRITICAL ERROR: Failed to prepare data. Server cannot start. Details: {e}") from e


@app.route('/recommend/<product_id>', methods=['GET'])
def recommend(product_id):
    """API endpoint to get recommendations for a given product_id."""
    print(f"\nReceived recommendation request for product: {product_id}") # Log request

    # Check if data structures are initialized (they should be after startup call)
    if product_to_idx is None or not isinstance(product_to_idx, dict):
         print("Error: product_to_idx map not initialized.")
         return jsonify({"error": "Recommendation engine is not ready (mapping error)."}), 503

    # Get optional parameter N from query string
    try:
        n_recommendations = request.args.get('N', default=10, type=int)
        if n_recommendations <= 0:
            raise ValueError("N must be a positive integer.")
    except (ValueError, TypeError):
        print(f"Invalid 'N' parameter received: {request.args.get('N')}")
        return jsonify({"error": "Invalid parameter 'N'. Must be a positive integer."}), 400

    # Check if product exists in our mapping
    if product_id not in product_to_idx:
        print(f"Product ID '{product_id}' not found in product map.")
        return jsonify({"error": f"Product '{product_id}' not found."}), 404

    # Check if at least one recommendation component is ready
    if not (similarity_ok or cf_buy_ok or cf_quote_ok):
         print("Warning: No recommendation components (similarity, cf_buy, cf_quote) are ready.")
         # Decide if this is an error or just returns empty list
         # Let's return empty list for now, as it might still be valid in some hybrid logic
         # return jsonify({"error": "Recommendation engine components failed initialization."}), 503

    start_rec_time = time.time()
    try:
        # Call the hybrid recommendation function
        recommendations_df = get_recommendations_hybrid_rerank(
            input_product=product_id,
            N=n_recommendations,
            # Default weights/params - consider making them configurable
            content_weight=0.3,
            cf_buy_weight=0.5,
            cf_quote_weight=0.2,
            k=1,
            fetch_top_M=50
        )
        end_rec_time = time.time()

        if recommendations_df is None or recommendations_df.empty:
            print(f"No recommendations generated for {product_id}.")
            # Return empty list instead of error, as it's a valid outcome
            response_data = {"product_id": product_id, "recommendations": []}
        else:
            # Convert DataFrame to list of dictionaries for JSON response
            # Ensure scores are JSON serializable (standard floats)
            recommendations_df['hybrid_score'] = recommendations_df['hybrid_score'].astype(float)
            result = recommendations_df.to_dict('records')
            print(f"Generated {len(result)} recommendations for {product_id} in {end_rec_time - start_rec_time:.4f}s.")
            response_data = {"product_id": product_id, "recommendations": result}

        return jsonify(response_data)

    except Exception as e:
        print(f"ERROR during recommendation generation for {product_id}: {e}")
        traceback.print_exc() # Log traceback for debugging
        return jsonify({"error": "An internal error occurred while generating recommendations."}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    # Check if essential components are loaded
    status = "ok"
    details = {}
    if product_to_idx is None or not product_to_idx:
        status = "error"
        details["product_map"] = "not initialized"
    if not (similarity_ok or cf_buy_ok or cf_quote_ok):
        status = "warning" # Or error depending on requirements
        details["recommendation_components"] = "one or more failed"
        details["status_flags"] = f"similarity={similarity_ok}, cf_buy={cf_buy_ok}, cf_quote={cf_quote_ok}"

    http_status = 200 if status == "ok" else 503 # Service Unavailable if critical parts failed

    return jsonify({"status": status, "details": details}), http_status

# --- Main Execution ---
if __name__ == '__main__':
    # --- Call the data loading and preparation function directly AT STARTUP---
    print("--- Initializing Recommendation Engine ---")
    load_and_prepare_data()
    # The function will raise SystemExit if critical errors occur, stopping here.
    print("--- Initialization Potentially Complete (Check Status Flags Above) ---")

    # --- Start the Flask server ---
    print(f"Starting Flask server on http://0.0.0.0:{FLASK_PORT}...")
    # Use host='0.0.0.0' to make it accessible externally (e.g., from Docker)
    # debug=False is recommended for production/stability
    # use_reloader=False prevents the setup code from running twice in debug mode
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, use_reloader=False)