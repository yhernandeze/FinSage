import pandas as pd
import numpy as np
import re
import argparse
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib


# Funciones de preprocesamiento

def clean_text_simple(text: str) -> str:
    """Limpieza ligera para titulares financieros."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9%\$\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_percent(x):
    """Convierte un porcentaje (string) a float."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip().replace("%", "")
    try:
        return float(x)
    except ValueError:
        return np.nan


def parse_number(x):
    """Convierte números con comas o texto a float."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        x = x.replace(",", "").strip()
    try:
        return float(x)
    except ValueError:
        return np.nan



# Función principal

def main(input_path: str, model_prefix: str):
    print(f"Cargando dataset original: {input_path}")
    df = pd.read_csv(input_path)
    print("Shape original:", df.shape)

    # Asegurarnos de que existen las columnas clave
    required_cols = ["Sentiment", "Headline"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Falta la columna requerida en el CSV: {c}")

    # Eliminar filas sin Sentiment o Headline
    df = df.dropna(subset=["Sentiment", "Headline"])
    print("Shape tras eliminar nulos clave:", df.shape)

    # Quedarnos solo con Positive / Negative
    df = df[df["Sentiment"].isin(["Positive", "Negative"])].copy()
    print("Shape tras filtrar Positive/Negative:", df.shape)
    print("Distribución de clases:")
    print(df["Sentiment"].value_counts())

    # Limpieza de texto
    print("Limpiando Headline...")
    df["Headline_clean"] = df["Headline"].apply(clean_text_simple)

    # Columnas numéricas
    print("Convirtiendo columnas numéricas...")
    if "Index_Change_Percent" in df.columns:
        df["Index_Change_Percent_num"] = df["Index_Change_Percent"].apply(parse_percent)
    else:
        df["Index_Change_Percent_num"] = np.nan

    if "Trading_Volume" in df.columns:
        df["Trading_Volume_num"] = df["Trading_Volume"].apply(parse_number)
    else:
        df["Trading_Volume_num"] = np.nan

    # Columnas categóricas: si no existen, se crean como NaN
    cat_cols = ["Sector", "Market_Event", "Market_Index", "Impact_Level", "Related_Company", "Source"]
    for col in cat_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Eliminar filas con texto vacío
    df = df[df["Headline_clean"].str.strip() != ""]
    print("Shape final tras limpieza:", df.shape)

    
    # Construcción de X e y
    
    text_col = "Headline_clean"
    numeric_cols = ["Index_Change_Percent_num", "Trading_Volume_num"]
    y = df["Sentiment"]
    X = df[[text_col] + numeric_cols + cat_cols]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Tamaño train: {X_train.shape[0]}, test: {X_test.shape[0]}")

    
    # Definición de transformadores
    
    # Texto -> TF-IDF (no tiene NaN porque ya limpiamos y reemplazamos nulos por "")
    text_transformer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=20000
    )

    # Numéricos: imputar + escalar
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categóricas: imputar + one-hot
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, text_col),
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    # Modelo
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])

    
    # Entrenamiento
    print("Entrenando modelo mixto (texto + numéricos + categóricos)...")
    model.fit(X_train, y_train)

    # Evaluación
    print("Evaluando en test...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("\n===== RESULTADOS MODELO MIXTO =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Macro:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    # Guardar modelo
    model_path = f"{model_prefix}_mixed_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nModelo mixto guardado en: {model_path}")


# Bloque de ejecución
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modelo mixto texto + num + cat para sentimiento binario.")

    parser.add_argument(
        "--input",
        type=str,
        default="financial_news_events.csv",
        help="Ruta al CSV original."
    )

    parser.add_argument(
        "--model-prefix",
        type=str,
        default="financial_sentiment_mixed",
        help="Prefijo para guardar el modelo entrenado."
    )

    args = parser.parse_args()
    main(args.input, args.model_prefix)