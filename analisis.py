#ANALISIS EXPLORATORIO DE LOS DATOS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import missingno as msno
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
os.chdir(r"D:\Documents\FUNCIONES\Tarea2CD")
df=pd.read_csv('bank-additional-full.csv',sep=';')



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
        'Valores Faltantes': missing_data,
        'Porcentaje': missing_percent
    }).sort_values('Porcentaje', ascending=False)
    
    print(missing_df[missing_df['Valores Faltantes'] > 0])
    
    # Visualización de missingness
    if missing_data.sum() > 0:
        plt.figure(figsize=(10, 6))
        msno.matrix(df)
        plt.title('Mapa de Datos Faltantes')
        plt.show()
        
        plt.figure(figsize=(10, 6))
        msno.heatmap(df)
        plt.title('Correlación de Faltantes')
        plt.show()
    
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




