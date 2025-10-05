import os
import uuid
from flask import Flask, render_template, request, redirect, flash, make_response
import pandas as pd
import numpy as np
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
import io
import csv

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'dev_secret')

# === Convenciones de nombres por dataset ===
# Modelos principales
#   - boost_{ds}.joblib  (preferido si existe)
#   - rf_{ds}.joblib     (fallback)
# Scalers (opcionales)
#   - scaler_{ds}.joblib
# Features
#   - feature_importances_{ds}.csv
#   - selected_features_{ds}_minimal.csv
# Samples (opcionales, para vista)
#   - static/sample_{ds}.csv

DATASETS = ['kepler', 'tess', 'k2']

# Nombres "globales" legacy (si no se pasa dataset)
RF_MODEL_PATH = 'rf_kepler.joblib'
SCALER_PATH = 'scaler_kepler.joblib'
FEATURES_CSV = 'feature_importances.csv'
LABEL_ENCODER_PATH = 'label_encoder_kepler.joblib'
TOP_K = int(os.environ.get('TOP_K_FEATURES', 20))

# Minimal (si existiera)
MIN_RF_MODEL_PATH = 'rf_kepler_minimal.joblib'
MIN_SCALER_PATH = 'scaler_kepler_minimal.joblib'
MIN_SELECTED_PATH = 'selected_features_minimal.csv'

# Cargas legacy (no se usan si se carga por dataset correctamente)
rf_model = None
scaler = None
expected_features = None
label_encoder = None
rf_min = None
scaler_min = None
required_features = None
use_minimal = False


# ================= Utilidades =================

def resolve_path(fname: str):
    """Busca fname en Rod/, Rod/.. y cwd. Devuelve ruta absoluta o None."""
    base_dir = os.path.dirname(__file__)  # Rod/
    cand1 = os.path.join(base_dir, fname)
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.join(os.path.dirname(base_dir), fname)
    if os.path.exists(cand2):
        return cand2
    cand3 = os.path.join(os.getcwd(), fname)
    if os.path.exists(cand3):
        return cand3
    return None


def detect_sep(path: str) -> str:
    """Heurística simple para detectar separador."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("#") and line.strip():
                    if "," in line and ";" in line:
                        return "," if line.count(",") > line.count(";") else ";"
                    elif ";" in line:
                        return ";"
                    elif "\t" in line:
                        return "\t"
                    else:
                        return ","
    except Exception:
        pass
    return ","


class NoOpScaler(TransformerMixin, BaseEstimator):
    """Scaler vacío para cuando no se tiene scaler entrenado."""
    def fit(self, X, y=None): return self
    def transform(self, X):   return X


def _read_csv_robusto(path: str) -> pd.DataFrame:
    """Lectura de CSV tolerante a archivos con comillas y una sola columna."""
    sep = detect_sep(path)
    df = pd.read_csv(path, sep=sep)
    # Si vino como una sola columna con comas dentro, repararlo
    if df.shape[1] == 1 and ("," in df.columns[0] or df.iloc[:, 0].astype(str).str.contains(",").any()):
        parts = df.iloc[:, 0].astype(str).str.split(',', expand=True)
        header = parts.iloc[0].tolist()
        parts = parts.iloc[1:].reset_index(drop=True)
        parts.columns = [str(h).strip() for h in header]
        df = parts
    df.columns = [c.strip() for c in df.columns]
    return df


def _write_template_file(dataset: str, cols: list, mode: str):
    """Escribe un CSV template en static/templates y devuelve ruta relativa."""
    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, 'static', 'templates')
    os.makedirs(out_dir, exist_ok=True)
    fname = f'template_{dataset}_{mode}.csv'
    out_path = os.path.join(out_dir, fname)
    try:
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerow(['' for _ in cols])
        return os.path.join('templates', fname)
    except Exception as e:
        print('Could not write template file', out_path, e)
        return None


def load_artifacts_for_dataset(dataset: str):
    """
    Carga artefactos para un dataset (kepler/tess/k2).
    Retorna dict con: rf (modelo; puede ser el boost), scaler, expected_features,
                      rf_min, scaler_min, required_features, pipeline_path
    """
    res = dict(
        rf=None, scaler=None, expected_features=None,
        rf_min=None, scaler_min=None, required_features=None,
        pipeline_path=None
    )
    if not dataset:
        return res

    ds = dataset.lower()
    boost_name = f'boost_{ds}.joblib'
    rf_name = f'rf_{ds}.joblib'
    scaler_name = f'scaler_{ds}.joblib'
    feat_name = f'feature_importances_{ds}.csv'
    sel_name = f'selected_features_{ds}_minimal.csv'

    # 1) Modelo: preferir boost si existe
    boost_path = resolve_path(boost_name)
    if boost_path:
        try:
            res['rf'] = joblib.load(boost_path)  # guardamos en 'rf' para unificar
            res['pipeline_path'] = boost_path
            # No retornamos todavía para que intentemos cargar scaler/features también
        except Exception as e:
            print(f'[ARTIFACTS] No se pudo cargar {boost_name}: {e}')

    # 2) RF fallback (si no hubo boost)
    if res['rf'] is None:
        rf_path = resolve_path(rf_name) or resolve_path(RF_MODEL_PATH)
        if rf_path:
            try:
                res['rf'] = joblib.load(rf_path)
                res['pipeline_path'] = rf_path
            except Exception as e:
                print(f'[ARTIFACTS] No se pudo cargar {rf_name}: {e}')

    # 3) Scaler (opcional)
    scaler_path = resolve_path(scaler_name) or resolve_path(SCALER_PATH)
    if scaler_path:
        try:
            res['scaler'] = joblib.load(scaler_path)
        except Exception as e:
            print(f'[ARTIFACTS] No se pudo cargar {scaler_name}: {e}')

    # 4) Feature importances (opcional)
    feat_path = resolve_path(feat_name) or resolve_path(FEATURES_CSV)
    if feat_path:
        try:
            fi = pd.read_csv(feat_path, index_col=0)
            try:
                importance_col = fi.columns[0]
                top_feats = fi.sort_values(by=importance_col, ascending=False).index.tolist()
            except Exception:
                top_feats = fi.index.tolist()
            res['expected_features'] = top_feats[:TOP_K] if TOP_K and TOP_K > 0 else top_feats
        except Exception as e:
            print(f'[ARTIFACTS] No se pudo leer {feat_name}: {e}')

    # 5) Minimal selected features (opcional)
    sel_path = resolve_path(sel_name) or resolve_path(MIN_SELECTED_PATH)
    if sel_path:
        try:
            sel = pd.read_csv(sel_path, header=0)
            res['required_features'] = sel['feature'].tolist()
        except Exception as e:
            print(f'[ARTIFACTS] No se pudo leer {sel_name}: {e}')

    # 6) Minimal models (opcional)
    if res['required_features']:
        min_rf = resolve_path(f'rf_{ds}_minimal.joblib') or resolve_path(MIN_RF_MODEL_PATH)
        min_scaler = resolve_path(f'scaler_{ds}_minimal.joblib') or resolve_path(MIN_SCALER_PATH)
        if min_rf:
            try:
                res['rf_min'] = joblib.load(min_rf)
                res['pipeline_path'] = min_rf
            except Exception as e:
                print(f'[ARTIFACTS] No se pudo cargar rf minimal: {e}')
        if min_scaler:
            try:
                res['scaler_min'] = joblib.load(min_scaler)
            except Exception as e:
                print(f'[ARTIFACTS] No se pudo cargar scaler minimal: {e}')

    return res


# ============== Cargas legacy (opcionales) ==============

rf_model_path = resolve_path(RF_MODEL_PATH)
if rf_model_path:
    try:
        rf_model = joblib.load(rf_model_path)
    except Exception as e:
        print('No se pudo cargar el modelo RF legacy:', e)

scaler_path = resolve_path(SCALER_PATH)
if scaler_path:
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print('No se pudo cargar el scaler legacy:', e)

features_csv_path = resolve_path(FEATURES_CSV)
if features_csv_path:
    try:
        fi = pd.read_csv(features_csv_path, index_col=0)
        try:
            importance_col = fi.columns[0]
            top_feats = fi.sort_values(by=importance_col, ascending=False).head(TOP_K).index.tolist()
        except Exception:
            top_feats = fi.index.tolist()[:TOP_K]
        expected_features = top_feats
        print(f'Usando top-{TOP_K} features legacy de {FEATURES_CSV}:', expected_features)
    except Exception as e:
        print('No se pudo leer feature_importances.csv legacy:', e)

label_encoder_path = resolve_path(LABEL_ENCODER_PATH)
if label_encoder_path:
    try:
        label_encoder = joblib.load(label_encoder_path)
    except Exception:
        label_encoder = None

min_selected_path = resolve_path(MIN_SELECTED_PATH)
if min_selected_path:
    try:
        sel = pd.read_csv(min_selected_path, header=0)
        required_features = sel['feature'].tolist()
        print('Required minimal features legacy loaded:', required_features)
    except Exception as e:
        print('Could not load selected_features_minimal.csv legacy:', e)

if required_features is not None:
    min_rf_path = resolve_path(MIN_RF_MODEL_PATH)
    min_scaler_path = resolve_path(MIN_SCALER_PATH)
    if min_rf_path and min_scaler_path:
        try:
            rf_min = joblib.load(min_rf_path)
            scaler_min = joblib.load(min_scaler_path)
            use_minimal = False  # por defecto apagado
            required_features = None
            print('Minimal artifacts legacy encontrados pero minimal-mode desactivado.')
        except Exception as e:
            print('Could not load minimal artifacts legacy:', e)


# ================= Rutas =================

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/classify/<dataset>', methods=['GET', 'POST'])
def classify_dataset(dataset='kepler'):
    dataset = (dataset or 'kepler').lower()
    if dataset not in DATASETS:
        flash(f'Dataset desconocido: {dataset}. Usa kepler/tess/k2')
        return redirect('/')

    artifacts = load_artifacts_for_dataset(dataset)

    if request.method == 'POST':
        if 'csv_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        csv_file = request.files['csv_file']
        if csv_file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        UUID = uuid.uuid4()
        save_dir = os.path.join('inputs')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{UUID}.csv')
        csv_file.save(save_path)
        use_min = 'use_minimal' in request.form and request.form.get('use_minimal') in ('1', 'true', 'on')
        use_min_val = '1' if use_min else '0'
        return redirect(f'/results/{UUID}?minimal={use_min_val}&dataset={dataset}')

    # Contexto de template: sample + columnas para previsualizar
    minimal_available = True if artifacts.get('rf_min') is not None and artifacts.get('required_features') is not None else False
    minimal_feature_count = len(artifacts.get('required_features') or [])
    full_features = artifacts.get('expected_features')
    sample_url = f"/static/sample_{dataset}.csv"

    columns, rows = None, None
    try:
        sample_path = resolve_path(f'sample_{dataset}.csv') or resolve_path(os.path.join('static', f'sample_{dataset}.csv'))
        if sample_path and os.path.exists(sample_path):
            sp = pd.read_csv(sample_path)
            columns = sp.columns.tolist()
            rows = sp.to_dict(orient='records')
    except Exception:
        columns, rows = None, None

    return render_template(
        'classify.html',
        features_required=(artifacts.get('required_features') if minimal_available else None),
        minimal_available=minimal_available,
        minimal_feature_count=minimal_feature_count,
        full_features=full_features,
        dataset=dataset,
        sample_url=sample_url,
        table_columns=columns,
        table_rows=rows
    )


@app.route('/download_template/<dataset>', methods=['GET'])
def download_template(dataset='kepler'):
    """Devuelve un CSV template para el dataset. mode: full|minimal."""
    dataset = (dataset or 'kepler').lower()
    if dataset not in DATASETS:
        return "Dataset desconocido", 404

    mode = request.args.get('mode', 'full')
    artifacts = load_artifacts_for_dataset(dataset)

    if mode == 'minimal' and artifacts.get('required_features'):
        cols = artifacts.get('required_features')
        filename = f'template_{dataset}_minimal.csv'
    else:
        cols = artifacts.get('expected_features') or []
        filename = f'template_{dataset}_full.csv'

    if not cols:
        cols = ['col1', 'col2']  # fallback mínimo

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(cols)
    writer.writerow(['' for _ in cols])
    content = buf.getvalue()
    buf.close()

    resp = make_response(content)
    resp.headers['Content-Type'] = 'text/csv; charset=utf-8'
    resp.headers['Content-Disposition'] = f'attachment; filename={filename}'
    return resp


@app.route('/results/<uuid:UUID>', methods=['GET'])
def results(UUID):
    # 1) Cargar CSV
    csv_path = f'inputs/{UUID}.csv'
    try:
        df = _read_csv_robusto(csv_path)
        if df is None or df.empty:
            flash('El CSV está vacío o no se pudo leer.')
            return render_template('results.html', df=pd.DataFrame())
    except Exception as e:
        print("Error al cargar el archivo:", e)
        flash(f'Error al cargar CSV: {e}')
        return render_template('results.html', df=pd.DataFrame())

    # 2) Dataset de query
    dataset = (request.args.get('dataset') or 'kepler').lower()
    if dataset not in DATASETS:
        dataset = 'kepler'

    # 3) Artefactos
    artifacts = load_artifacts_for_dataset(dataset)

    # 4) Modelo + scaler
    model = artifacts.get('rf') or rf_model  # 'rf' puede ser un boost mapeado
    if model is None:
        flash(f'Modelo no encontrado para {dataset}. (¿Existe boost_{dataset}.joblib o rf_{dataset}.joblib?)')
        return render_template('results.html', df=df)

    scaler_local = artifacts.get('scaler') or scaler or NoOpScaler()

    # 5) Features en orden
    features = None
    try:
        fnames = getattr(model, 'feature_names_in_', None)
        if fnames is not None and len(fnames) > 0:
            features = list(fnames)
    except Exception:
        pass
    if not features:
        features = artifacts.get('expected_features')
    if not features:
        # último recurso: columnas numéricas del CSV
        features = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # 6) Rellenar faltantes y ordenar
    missing = [c for c in features if c not in df.columns]
    if missing:
        med = df.select_dtypes(include=[np.number]).median()
        for c in missing:
            df[c] = med.mean() if med.shape[0] > 0 else 0.0
        flash(f'Columnas faltantes rellenadas: {missing}')

    X = df.reindex(columns=features).apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median().fillna(0.0))

    # 7) Validar dimensión
    try:
        n_in = getattr(model, 'n_features_in_', None)
        if n_in is not None and X.shape[1] != n_in:
            flash(f'Número de features ({X.shape[1]}) no coincide con lo esperado ({n_in}).')
            return render_template('results.html', df=df)
    except Exception:
        pass

    # 8) Escalar y predecir
    try:
        try:
            Xs = scaler_local.transform(X)
        except Exception:
            Xs = X.values
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(Xs)
            proba_pos = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else None
        else:
            proba_pos = None
        preds = model.predict(Xs)
    except Exception as e:
        flash(f'Error durante la predicción: {e}')
        return render_template('results.html', df=df)

    # 9) Salida
    df_display = df.copy()
    df_display['pred_is_exoplanet'] = preds
    if proba_pos is not None:
        df_display['prob_is_exoplanet'] = proba_pos

    # Reordenar columnas (opcional)
    target_cols = [c for c in df_display.columns if c.lower() in ('is_exoplanet', 'isplanet', 'true_label', 'label')]
    non_targets = [c for c in df_display.columns if c not in target_cols and c not in ('prob_is_exoplanet', 'pred_is_exoplanet')]
    new_order = non_targets + target_cols + [c for c in ['prob_is_exoplanet', 'pred_is_exoplanet'] if c in df_display.columns]
    df_display = df_display.reindex(columns=[c for c in new_order if c in df_display.columns])

    # Conteo
    try:
        summary = df_display['pred_is_exoplanet'].value_counts().to_dict()
    except Exception:
        summary = {}

    return render_template('results.html',
                           df=df_display,
                           summary=summary,
                           features_used=features,
                           pipeline_used=artifacts.get('pipeline_path') or 'modelo')


# ============== Generación de templates al inicio ==============

def _generate_startup_templates():
    for ds in DATASETS:
        try:
            art = load_artifacts_for_dataset(ds)
            # FULL
            cols_full = art.get('expected_features') or []
            if not cols_full:
                # intentar con sample
                sample_path = resolve_path(os.path.join('static', f'sample_{ds}.csv')) or resolve_path(f'sample_{ds}.csv')
                if sample_path and os.path.exists(sample_path):
                    try:
                        _df = pd.read_csv(sample_path, nrows=0)
                        cols_full = _df.columns.tolist()
                    except Exception:
                        cols_full = []
            if cols_full:
                _write_template_file(ds, cols_full, 'full')
            # MINIMAL
            cols_min = art.get('required_features') or []
            if cols_min:
                _write_template_file(ds, cols_min, 'minimal')
        except Exception as e:
            print('Template generation error for', ds, e)

# Ejecutar al importar
_generate_startup_templates()


if __name__ == '__main__':
    app.run(debug=True)
