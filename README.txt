# K2 – Modelo Físico para Detección de Exoplanetas

Este proyecto implementa un **modelo de clasificación de exoplanetas** basado en datos del catálogo **K2** (NASA).  
Utiliza **Random Forest** con atributos físicos de los planetas y estrellas, y una **interfaz web Flask** para visualizar ejemplos de datos y predicciones.

---

## Características

- Entrenamiento de un modelo **Boosting** sobre variables físicas (sin metadatos post-análisis).  
- Exportación del modelo entrenado (`.joblib`) listo para predicción.  
- Visualización web con Flask:
  - Vista `/classify/<dataset>` que muestra una **tabla de ejemplo** (`sample_<dataset>.csv`).  
  - Tabla con **scroll horizontal y vertical**, compatible con catálogos grandes.  
- Base lista para extenderse a **Kepler**, **TESS** u otros catálogos.

---

## Estructura del proyecto

Proyecto NASA
├── app.py # Servidor Flask con vista de clasificación
├── train_k2_rf.py # Entrenamiento del modelo Random Forest (K2)
├── k2_rf_physical.joblib # Modelo entrenado (generado tras entrenamiento)
├── templates/
│ └── classify.html # Plantilla HTML con scroll para tabla de muestra
├── static/
│ └── sample_K2.csv # Ejemplo de dataset (vista previa)
├── requirements.txt # Dependencias del proyecto
└── README.md # Este archivo