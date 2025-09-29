#ANALISIS EXPLORATORIO DE LOS DATOS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import missingno as msno
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.linalg import inv









# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
os.chdir(r"D:\Documents\FUNCIONES\Tarea2CD")
df=pd.read_csv('bank-additional-full.csv',sep=';')
print(df)


def eda_completo(df, target_var=None):
    """
    EDA completo automatizado para cualquier dataset
    """
    print("="*50)
    print("ANÁLISIS EXPLORATORIO COMPLETO (EDA)")
    print("="*50)
    
    # 1. INFORMACIÓN BÁSICA DEL DATASET
    print("\n1. INFORMACIÓN BÁSICA")
    print(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"Memoria usada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 2. TIPOS DE DATOS
    print("\n2. TIPOS DE DATOS")
    print(df.dtypes.value_counts())
    # 3. DATOS FALTANTES
print("\n3. ANÁLISIS DE DATOS FALTANTES")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100

missing_df = pd.DataFrame({
    'Valores_Faltantes': missing_data,
    'Porcentaje': missing_percent
}).sort_values('Porcentaje', ascending=False)

# FILTRAR solo variables con faltantes (evita mostrar todas)
missing_with_values = missing_df[missing_df['Valores_Faltantes'] > 0]

if not missing_with_values.empty:
    print("Variables con datos faltantes:")
    print(missing_with_values)
    
    # Visualización de missingness
    plt.figure(figsize=(12, 6))
    msno.matrix(df)
    plt.title('Mapa de Datos Faltantes')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    msno.heatmap(df)
    plt.title('Correlación de Datos Faltantes')
    plt.tight_layout()
    plt.show()
    
    # Estadísticas adicionales útiles
    print(f"\nResumen de datos faltantes:")
    print(f"Total de valores faltantes: {missing_data.sum()}")
    print(f"Porcentaje total de faltantes: {(missing_data.sum() / (len(df) * len(df.columns)) * 100):.2f}%")
    print(f"Variables con faltantes: {len(missing_with_values)} de {len(df.columns)}")
    
else:
    print("✅ No se encontraron datos faltantes en el dataset.")
    # 4. ESTADÍSTICAS DESCRIPTIVAS
    print("\n4. ESTADÍSTICAS DESCRIPTIVAS")
    print(df.describe(include='all').T)
    
    # 5. ANÁLISIS UNIVARIADO
    print("\n5. ANÁLISIS UNIVARIADO")
    
    # Variables numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nVariables numéricas ({len(numeric_cols)}): {list(numeric_cols)}")
        
        # Gráficos para cada variable numérica
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*4))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                # Histograma + KDE
                df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                df[col].plot.kde(ax=axes[i], secondary_y=True)
                axes[i].set_title(f'Distribución de {col}')
                axes[i].set_xlabel(col)
        
        # Ocultar ejes vacíos
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        plt.show()
        
        # Boxplots para detectar outliers
        if len(numeric_cols) > 0:
            n_cols_box = min(4, len(numeric_cols))
            n_rows_box = (len(numeric_cols) + n_cols_box - 1) // n_cols_box
            
            fig, axes = plt.subplots(n_rows_box, n_cols_box, figsize=(15, n_rows_box*4))
            axes = axes.flatten() if n_rows_box > 1 else [axes]
            
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    df.boxplot(column=col, ax=axes[i])
                    axes[i].set_title(f'Boxplot de {col}')
            
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
                
            plt.tight_layout()
            plt.show()
    
    # Variables categóricas
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"\nVariables categóricas ({len(categorical_cols)}): {list(categorical_cols)}")
        
        n_cols_cat = min(2, len(categorical_cols))
        n_rows_cat = (len(categorical_cols) + n_cols_cat - 1) // n_cols_cat
        
        fig, axes = plt.subplots(n_rows_cat, n_cols_cat, figsize=(15, n_rows_cat*5))
        axes = axes.flatten() if n_rows_cat > 1 else [axes]
        
        for i, col in enumerate(categorical_cols):
            if i < len(axes):
                top_categories = df[col].value_counts().head(10)
                top_categories.plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'Top 10 categorías de {col}')
                axes[i].tick_params(axis='x', rotation=45)
        
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    
    # 6. ANÁLISIS BIVARIADO/MULTIVARIADO
    print("\n6. ANÁLISIS DE RELACIONES")
    
    # Matriz de correlación (solo numéricas)
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        plt.show()
        
        # Scatter matrix
        if len(numeric_cols) <= 6:  # No mostrar si hay muchas variables
            pd.plotting.scatter_matrix(df[numeric_cols], alpha=0.6, 
                                     figsize=(12, 12), diagonal='hist')
            plt.suptitle('Matriz de Scatter Plots', y=0.95)
            plt.show()
    
    # 7. ANÁLISIS CON VARIABLE TARGET (si existe)
    if target_var and target_var in df.columns:
        print(f"\n7. ANÁLISIS CON VARIABLE TARGET: {target_var}")
        
        # Si el target es numérico
        if df[target_var].dtype in [np.number]:
            # Correlación con target
            target_corr = df[numeric_cols].corrwith(df[target_var]).sort_values(ascending=False)
            print("Correlación con target:")
            print(target_corr)
            
            # Gráficos de relación con target
            n_cols_target = min(3, len(numeric_cols) - 1)  # Excluir target
            n_rows_target = ((len(numeric_cols) - 1) + n_cols_target - 1) // n_cols_target
            
            if n_rows_target > 0:
                fig, axes = plt.subplots(n_rows_target, n_cols_target, figsize=(15, n_rows_target*4))
                axes = axes.flatten() if n_rows_target > 1 else [axes]
                
                features = [col for col in numeric_cols if col != target_var]
                for i, col in enumerate(features):
                    if i < len(axes):
                        axes[i].scatter(df[col], df[target_var], alpha=0.6)
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel(target_var)
                        axes[i].set_title(f'{col} vs {target_var}')
                
                for j in range(i+1, len(axes)):
                    axes[j].set_visible(False)
                    
                plt.tight_layout()
                plt.show()
        
        # Si el target es categórico
        elif df[target_var].dtype in ['object', 'category']:
            # Boxplots por categoría del target
            if len(numeric_cols) > 0:
                n_cols_target = min(3, len(numeric_cols))
                n_rows_target = (len(numeric_cols) + n_cols_target - 1) // n_cols_target
                
                fig, axes = plt.subplots(n_rows_target, n_cols_target, figsize=(15, n_rows_target*4))
                axes = axes.flatten() if n_rows_target > 1 else [axes]
                
                for i, col in enumerate(numeric_cols):
                    if i < len(axes):
                        df.boxplot(column=col, by=target_var, ax=axes[i])
                        axes[i].set_title(f'{col} por {target_var}')
                
                for j in range(i+1, len(axes)):
                    axes[j].set_visible(False)
                    
                plt.tight_layout()
                plt.show()
    
    # 8. RESUMEN FINAL
    print("\n8. RESUMEN Y RECOMENDACIONES")
    print(f"• Variables numéricas: {len(numeric_cols)}")
    print(f"• Variables categóricas: {len(categorical_cols)}")
    print(f"• Total de valores faltantes: {df.isnull().sum().sum()}")
    
    if missing_data.sum() > 0:
        print("• RECOMENDACIÓN: Considerar estrategias para datos faltantes")
    
    # Detección de posibles problemas
    high_cardinality = [col for col in categorical_cols if df[col].nunique() > 50]
    if high_cardinality:
        print(f"• ALERTA: Variables con alta cardinalidad: {high_cardinality}")
    
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        print(f"• ALERTA: Variables constantes: {constant_cols}")

eda_completo(df, target_var='target')

























#################################
#MODELO DE CLASIFICACIÓN 
# Separar train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print("Tamaño entrenamiento:", X_train.shape)
print("Tamaño prueba:", X_test.shape)



#=====================================
#======  Naive Bayes =================
#=====================================
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Entrenar
nb = GaussianNB()
nb.fit(X_train, y_train)
# Después del entrenamiento, puedes ver los priors:
print("Priors usados por el modelo:", nb.class_prior_)
print("Clases:", nb.classes_)
#Caso de aprioris iguales
#nb = GaussianNB(priors=[0.5, 0.5])  # Asumiendo [clase_0, clase_1]
#nb.fit(X_train, y_train)
# Para ver las aprioris
print("Distribución de clases en y_train:")
print(pd.Series(y_train).value_counts(normalize=True))

# Evaluar
y_pred_nb = nb.predict(X_test)
cm = confusion_matrix(y_test, y_pred_nb)
print("Matriz de confusión (Naive Bayes):\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Precisión:", precision_score(y_test, y_pred_nb, average="weighted"))
print("Sensibilidad:", recall_score(y_test, y_pred_nb, average="weighted"))
print("F1-score:", f1_score(y_test, y_pred_nb, average="weighted"))





#=====================================
#=LDA (Linear Discriminant Analysis)=
#=====================================

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Entrenar
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)


# Evaluar
y_pred_lda = lda.predict(X_test)
cm = confusion_matrix(y_test, y_pred_lda)
print("Matriz de confusión (LDA):\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred_lda))
print("Precisión:", precision_score(y_test, y_pred_lda, average='weighted'))
print("Sensibilidad:", recall_score(y_test, y_pred_lda, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred_lda, average='weighted'))

#=====================================
# QDA (Quadratic Discriminant Analysis) =
#=====================================
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Entrenar
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)


# Evaluar
y_pred_qda = qda.predict(X_test)
cm = confusion_matrix(y_test, y_pred_qda)
print("Matriz de confusión (QDA):\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred_qda))
print("Precisión:", precision_score(y_test, y_pred_qda, average='weighted'))
print("Sensibilidad:", recall_score(y_test, y_pred_qda, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred_qda, average='weighted'))

#=====================================
# Fisher
#=====================================

def fisher_direction(X, y):
    # Separar clases
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("Fisher solo para 2 clases")
    
    X0 = X[y == classes[0]]
    X1 = X[y == classes[1]]
    
    # Medias de cada clase
    mu0 = np.mean(X0, axis=0)
    mu1 = np.mean(X1, axis=0)
    
    # Matrices de covarianza dentro de cada clase
    S0 = np.cov(X0, rowvar=False, bias=True) * len(X0)
    S1 = np.cov(X1, rowvar=False, bias=True) * len(X1)
    
    # Matriz de dispersión de ambas clases
    SW = S0 + S1
    
    # Dirección de Fisher (sin el threshold)
    w = inv(SW) @ (mu1 - mu0)
    
    return w

# Obtener dirección de Fisher 
w_fisher = fisher_direction(X_train, y_train)

def umbral_fisher(projections, y_train):
    
    # Separar proyecciones por clase
    proj_class0 = projections[y_train == 0]
    proj_class1 = projections[y_train == 1]
    
    # Encontrar threshold que maximiza separación
    # Simple: punto medio entre medias proyectadas
    umbral = (np.mean(proj_class0) + np.mean(proj_class1)) / 2
    
    return umbral

# Encontrar umbral óptimo
umbral = umbral_fisher(X_train, y_train)


#Proyecta la prueba
X_test_fisher=umbral.T X_test
# Clasificar
y_pred_fisher = (X_test_fisher > umbral).astype(int)



#=====================================
# k-NN (k-Nearest Neighbors) =
#=====================================
from sklearn.neighbors import KNeighborsClassifier

# Entrenar (con k=5 vecinos)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# Evaluar
y_pred_knn = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred_knn)
print("Matriz de confusión (k-NN):\n", cm)
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Precisión:", precision_score(y_test, y_pred_knn, average='weighted'))
print("Sensibilidad:", recall_score(y_test, y_pred_knn, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred_knn, average='weighted'))


#=====================================
# Comparación de todos los modelos
#=====================================
import pandas as pd

# Guardar resultados en un diccionario
results = {
    "Naive Bayes": {
        "Acc": accuracy_score(y_test, y_pred_nb),
        "Precisión": precision_score(y_test, y_pred_nb, average="weighted"),
        "Recall": recall_score(y_test, y_pred_nb, average="weighted"),
        "F1": f1_score(y_test, y_pred_nb, average="weighted")
    },
    "LDA": {
        "Acc": accuracy_score(y_test, y_pred_lda),
        "Precisión": precision_score(y_test, y_pred_lda, average="weighted"),
        "Recall": recall_score(y_test, y_pred_lda, average="weighted"),
        "F1": f1_score(y_test, y_pred_lda, average="weighted")
    },
    "QDA": {
        "Acc": accuracy_score(y_test, y_pred_qda),
        "Precisión": precision_score(y_test, y_pred_qda, average="weighted"),
        "Recall": recall_score(y_test, y_pred_qda, average="weighted"),
        "F1": f1_score(y_test, y_pred_qda, average="weighted")
    },
    "k-NN (k=5)": {
        "Acc": accuracy_score(y_test, y_pred_knn),
        "Precisión": precision_score(y_test, y_pred_knn, average="weighted"),
        "Recall": recall_score(y_test, y_pred_knn, average="weighted"),
        "F1": f1_score(y_test, y_pred_knn, average="weighted")
    }
}

# Convertir a DataFrame para visualización
df_results = pd.DataFrame(results).T
print("\n=== Comparación Final de Modelos ===")
print(df_results.round(3))

#=====================================
# Validación cruzada comparativa
#=====================================

from sklearn.model_selection import StratifiedKFold, cross_val_score

models = {
    "Naive Bayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== Validación Cruzada (5-fold, accuracy) ===")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"{name:15s}: Acc = {scores.mean():.3f} ± {scores.std():.3f}")



#=====================================
# Matrices de confusión
#=====================================

import seaborn as sns

# Lista de modelos ya entrenados
trained_models = {
    "Naive Bayes": nb,
    "LDA": lda,
    "QDA": qda,
    "k-NN": knn
}

# Graficar matrices de confusión
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()
target_names = ['no', 'yes']  # Los valores únicos de tu columna 'y'
for ax, (name, model) in zip(axes, trained_models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")

plt.tight_layout()
plt.show()














