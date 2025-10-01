"""
29 de septiembre del 2025
Tarea 2 Introduccion a ciencia de datos
Rodrigo Jesus Cesar
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import multivariate_normal
from tqdm import tqdm#Tarda demasiado en ejecutar asi que esta paqueteria ayuda a ver el progreso
import warnings
warnings.filterwarnings('ignore')

class BayesianClassifier(BaseEstimator, ClassifierMixin):
    """Clasificador Bayes Optimo para distribuciones normales multivariadas"""
    
    def __init__(self, pi0=0.5, pi1=0.5, mu0=None, mu1=None, sigma0=None, sigma1=None):
        self.pi0 = pi0
        self.pi1 = pi1
        self.mu0 = mu0
        self.mu1 = mu1
        self.sigma0 = sigma0
        self.sigma1 = sigma1
    
    def fit(self, X, y):
        return self
    
    def predict_proba(self, X):
        """Calculo probabilidades posteriores"""
        f0 = multivariate_normal(self.mu0, self.sigma0).pdf(X)
        f1 = multivariate_normal(self.mu1, self.sigma1).pdf(X)
        denom = self.pi0 * f0 + self.pi1 * f1
        p0 = (self.pi0 * f0) / denom
        p1 = (self.pi1 * f1) / denom
        return np.column_stack([p0, p1])
    
    def predict(self, X):
        """Predice clases"""
        probas = self.predict_proba(X)
        return (probas[:, 1] > probas[:, 0]).astype(int)

class FisherClassifier(BaseEstimator, ClassifierMixin):
    """Clasificador de Fisher (proyección 1D) compatible con scikit-learn"""
    
    def __init__(self):
        self.w = None
        self.threshold = None
    
    def fit(self, X, y):
        X0 = X[y == 0]
        X1 = X[y == 1]
        mu0 = np.mean(X0, axis=0)
        mu1 = np.mean(X1, axis=0)
        
        n0, n1 = len(X0), len(X1)
        S0 = np.cov(X0, rowvar=False)
        S1 = np.cov(X1, rowvar=False)
        S_pooled = ((n0 - 1) * S0 + (n1 - 1) * S1) / (n0 + n1 - 2)
        
        self.w = np.linalg.solve(S_pooled, mu1 - mu0)
        
        z0 = X0 @ self.w
        z1 = X1 @ self.w
        self.threshold = (np.mean(z0) + np.mean(z1)) / 2
        
        return self
    
    def predict(self, X):
        projections = X @ self.w
        return (projections > self.threshold).astype(int)
    
    def get_params(self, deep=True):
        return {}
    
    def set_params(self, **parameters):
        return self

def calculate_true_risk(classifier, pi0, pi1, mu0, mu1, sigma0, sigma1, n_test=10000):
    """Calcula el riesgo verdadero mediante integración numérica"""
    n0_test = int(n_test * pi0)
    n1_test = int(n_test * pi1)
    
    X0_test = np.random.multivariate_normal(mu0, sigma0, n0_test)
    X1_test = np.random.multivariate_normal(mu1, sigma1, n1_test)
    X_test = np.vstack([X0_test, X1_test])
    y_test = np.hstack([np.zeros(n0_test), np.ones(n1_test)])
    
    y_pred = classifier.predict(X_test)
    error = 1 - accuracy_score(y_test, y_pred)
    
    return error

def manual_cross_validation(classifier, X, y, n_splits=5):
    """Validación cruzada manual para clasificadores personalizados"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    errors = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = classifier.__class__()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        error = 1 - accuracy_score(y_test, y_pred)
        errors.append(error)
    
    return np.mean(errors)

#==========Diseño experimental sugerido (flexibe)
def create_scenarios():
    """
    Define los diferentes escenarios de simulación como lo requiere la tarea
    """
    scenarios = {
        'Escenario 1: Covarianzas Iguales (LDA optimo)': {
            'mu0': np.array([-1.5, -1.5]),
            'mu1': np.array([1.5, 1.5]),
            'sigma0': np.array([[2.0, 0.8], [0.8, 1.5]]),
            'sigma1': np.array([[2.0, 0.8], [0.8, 1.5]]),
            'pi0': 0.5,
            'pi1': 0.5,
            'optimal': 'LDA'
        },
        'Escenario 2: Covarianzas Diferentes (QDA optimo)': {
            'mu0': np.array([-1.0, -1.0]),
            'mu1': np.array([1.0, 1.0]),
            'sigma0': np.array([[1.0, 0.2], [0.2, 1.0]]),
            'sigma1': np.array([[3.0, 1.2], [1.2, 2.0]]),
            'pi0': 0.5,
            'pi1': 0.5,
            'optimal': 'QDA'
        },
        'Escenario 3: Clases Desbalanceadas': {
            'mu0': np.array([-1.0, -1.0]),
            'mu1': np.array([1.0, 1.0]),
            'sigma0': np.array([[1.5, 0.5], [0.5, 1.5]]),
            'sigma1': np.array([[1.5, 0.5], [0.5, 1.5]]),
            'pi0': 0.8,
            'pi1': 0.2,
            'optimal': 'LDA'
        },
        'Escenario 4: Correlaciones Fuertes y Mal Condicionamiento': {
            'mu0': np.array([-0.5, -0.5]),
            'mu1': np.array([0.5, 0.5]),
            'sigma0': np.array([[2.0, 1.8], [1.8, 2.0]]),  # Alta correlación
            'sigma1': np.array([[2.0, -1.8], [-1.8, 2.0]]), # Alta correlación negativa
            'pi0': 0.5,
            'pi1': 0.5,
            'optimal': 'QDA'
        }
    }
    return scenarios
#=======================

#=====Programa para la simulacion=======
def run_simulation():
    """Ejecuta la simulación con los barridos de parámetros"""
    
    np.random.seed(42)
    
    # Parámetros del diseño experimental
    n_samples_list = [50, 100, 200, 500]  # por clase
    k_list = [1, 3, 5, 11, 21]  # para k-NN
    R = 20  # número de réplicas
    
    scenarios = create_scenarios()
    all_results = []
    
    for scenario_name, params in scenarios.items():
        print(f"\n{'='*60}")
        print(f"Ejecutando: {scenario_name}")
        print(f"{'='*60}")
        
        mu0, mu1 = params['mu0'], params['mu1']
        sigma0, sigma1 = params['sigma0'], params['sigma1']
        pi0, pi1 = params['pi0'], params['pi1']
        
        # Riesgo de Bayes optimo 
        bayes_classifier = BayesianClassifier(pi0, pi1, mu0, mu1, sigma0, sigma1)
        bayes_risk = calculate_true_risk(bayes_classifier, pi0, pi1, mu0, mu1, sigma0, sigma1)
        
        for n in n_samples_list:
            
            for replica in tqdm(range(R), desc=f"Réplicas"):
                # Generar datos
                n0 = int(n * pi0)
                n1 = int(n * pi1)
                
                X0 = np.random.multivariate_normal(mu0, sigma0, n0)
                X1 = np.random.multivariate_normal(mu1, sigma1, n1)
                X = np.vstack([X0, X1])
                y = np.hstack([np.zeros(n0), np.ones(n1)])
                
                # Métodos a comparar
                methods = {
                    'LDA': LinearDiscriminantAnalysis(),
                    'QDA': QuadraticDiscriminantAnalysis(),
                    'NaiveBayes': GaussianNB(),
                    'Fisher': FisherClassifier()
                }
                
                # Entrenar y evaluar métodos principales
                for method_name, classifier in methods.items():
                    # Entrenar
                    classifier.fit(X, y)
                    
                    # Riesgo verdadero
                    true_risk = calculate_true_risk(classifier, pi0, pi1, mu0, mu1, sigma0, sigma1)
                    
                    # Riesgo por validación cruzada
                    if method_name == 'Fisher':
                        cv_risk = manual_cross_validation(classifier, X, y)
                    else:
                        cv_risk = 1 - np.mean(cross_val_score(classifier, X, y, cv=5, scoring='accuracy'))
                    
                    # Brecha respecto a Bayes
                    gap = true_risk - bayes_risk
                    
                    # Almacenar resultados
                    all_results.append({
                        'Scenario': scenario_name,
                        'Method': method_name,
                        'n': n,
                        'k': None,
                        'Replica': replica,
                        'True_Risk': true_risk,
                        'CV_Risk': cv_risk,
                        'Bayes_Risk': bayes_risk,
                        'Gap': gap,
                        'Optimal': params['optimal']
                    })
                
                # Evaluar k-NN para diferentes valores de k
                for k in k_list:
                    knn = KNeighborsClassifier(n_neighbors=k)
                    knn.fit(X, y)
                    
                    # Riesgo verdadero
                    true_risk = calculate_true_risk(knn, pi0, pi1, mu0, mu1, sigma0, sigma1)
                    
                    # Riesgo por validación cruzada
                    cv_risk = 1 - np.mean(cross_val_score(knn, X, y, cv=5, scoring='accuracy'))
                    
                    # Brecha respecto a Bayes
                    gap = true_risk - bayes_risk
                    
                    # Almacenar resultados
                    all_results.append({
                        'Scenario': scenario_name,
                        'Method': f'kNN_k{k}',
                        'n': n,
                        'k': k,
                        'Replica': replica,
                        'True_Risk': true_risk,
                        'CV_Risk': cv_risk,
                        'Bayes_Risk': bayes_risk,
                        'Gap': gap,
                        'Optimal': params['optimal']
                    })
                
    return pd.DataFrame(all_results)
#======================================

#==========Generadora de graficas==============
def plot_results(df):
    """Generadora de graciass"""
    
    # Configuración de estilo
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. L(g) vs. n (por método)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    scenarios = df['Scenario'].unique()
    main_methods = ['LDA', 'QDA', 'NaiveBayes', 'Fisher']
    
    for idx, scenario in enumerate(scenarios):
        if idx >= 4:
            break
            
        ax = axes[idx]
        scenario_data = df[df['Scenario'] == scenario]
        
        for method in main_methods:
            method_data = scenario_data[scenario_data['Method'] == method]
            summary = method_data.groupby('n')['True_Risk'].agg(['mean', 'std']).reset_index()
            
            ax.plot(summary['n'], summary['mean'], 'o-', label=method, linewidth=2, markersize=6)
            ax.fill_between(summary['n'], 
                          summary['mean'] - summary['std'], 
                          summary['mean'] + summary['std'], 
                          alpha=0.2)
        
        # Línea de Bayes
        bayes_risk = scenario_data['Bayes_Risk'].iloc[0]
        ax.axhline(y=bayes_risk, color='black', linestyle='--', linewidth=2, 
                  label='Bayes (Optimo)')
        
        ax.set_xlabel('Tamaño muestral por clase (n)')
        ax.set_ylabel('Riesgo verdadero L(g)')
        ax.set_title(f'{scenario}\nEvolución del riesgo vs. n', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    #plt.savefig('1_riesgo_vs_n.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. L(k-NN) vs. k (curvas por n)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, scenario in enumerate(scenarios):
        if idx >= 4:
            break
            
        ax = axes[idx]
        scenario_data = df[df['Scenario'] == scenario]
        knn_data = scenario_data[scenario_data['Method'].str.startswith('kNN')]
        
        for n in [50, 100, 200, 500]:
            n_data = knn_data[knn_data['n'] == n]
            k_values = sorted(n_data['k'].unique())
            
            risks = []
            stds = []
            for k in k_values:
                k_data = n_data[n_data['k'] == k]
                risks.append(k_data['True_Risk'].mean())
                stds.append(k_data['True_Risk'].std())
            
            ax.plot(k_values, risks, 'o-', label=f'n={n}', linewidth=2, markersize=6)
        
        bayes_risk = scenario_data['Bayes_Risk'].iloc[0]
        ax.axhline(y=bayes_risk, color='black', linestyle='--', linewidth=2, 
                  label='Bayes (Optimo)')
        
        ax.set_xlabel('Número de vecinos (k)')
        ax.set_ylabel('Riesgo verdadero L(k-NN)')
        ax.set_title(f'{scenario}\nRiesgo k-NN vs. k', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    #plt.savefig('2_knn_vs_k.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Brechas L(g) - L(Bayes) vs. n (heatmaps y líneas)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, scenario in enumerate(scenarios):
        if idx >= 4:
            break
            
        ax = axes[idx]
        scenario_data = df[df['Scenario'] == scenario]
        
        # Preparar datos para heatmap
        methods = ['LDA', 'QDA', 'NaiveBayes', 'Fisher']
        n_values = sorted(scenario_data['n'].unique())
        
        gap_matrix = []
        for method in methods:
            method_gaps = []
            for n in n_values:
                method_n_data = scenario_data[(scenario_data['Method'] == method) & 
                                            (scenario_data['n'] == n)]
                gap_mean = method_n_data['Gap'].mean()
                method_gaps.append(gap_mean)
            gap_matrix.append(method_gaps)
        
        gap_matrix = np.array(gap_matrix)
        
        # Heatmap
        im = ax.imshow(gap_matrix, cmap='RdBu_r', aspect='auto', 
                      vmin=-0.1, vmax=0.1)
        
        # Configurar ejes
        ax.set_xticks(range(len(n_values)))
        ax.set_xticklabels(n_values)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        
        # Anotar valores
        for i in range(len(methods)):
            for j in range(len(n_values)):
                text = ax.text(j, i, f'{gap_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_xlabel('Tamaño muestral (n)')
        ax.set_ylabel('Método')
        ax.set_title(f'{scenario}\nBrecha L(g) - L(Bayes)', fontweight='bold')
        
        # Barra de color
        plt.colorbar(im, ax=ax, label='Brecha respecto a Bayes')
    
    plt.tight_layout()
    #plt.savefig('3_brechas_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Comparación validación vs. Monte Carlo
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, scenario in enumerate(scenarios):
        if idx >= 4:
            break
            
        ax = axes[idx]
        scenario_data = df[df['Scenario'] == scenario]
        
        methods = ['LDA', 'QDA', 'NaiveBayes', 'Fisher']
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            method_data = scenario_data[scenario_data['Method'] == method]
            summary = method_data.groupby('n').agg({
                'True_Risk': ['mean', 'std'],
                'CV_Risk': ['mean', 'std']
            }).reset_index()
            
            n_values = summary['n']
            true_mean = summary[('True_Risk', 'mean')]
            cv_mean = summary[('CV_Risk', 'mean')]
            
            ax.plot(n_values, true_mean, 'o-', color=colors[i], 
                   label=f'{method} (Verdadero)', linewidth=2)
            ax.plot(n_values, cv_mean, 's--', color=colors[i], 
                   label=f'{method} (CV)', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Tamaño muestral (n)')
        ax.set_ylabel('Riesgo')
        ax.set_title(f'{scenario}\nValidación vs Monte Carlo', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    #plt.savefig('4_validacion_vs_montecarlo.png', dpi=300, bbox_inches='tight')
    plt.show()
#==============================================

def generate_summary_tables(df):
    """Genera tablas resumen de resultados"""
        
    # Tabla por escenario y método
    summary_table = df.groupby(['Scenario', 'Method', 'n'])['True_Risk'].agg(['mean', 'std']).round(4)
    
    # Formatear como media +- std
    summary_table['Risk'] = summary_table.apply(lambda x: f"{x['mean']:.4f} +- {x['std']:.4f}", axis=1)
    summary_pivot = summary_table['Risk'].unstack('n')
    
    print("\nRiesgo verdadero por método, escenario y tamaño muestral:")
    print(summary_pivot)
    
    # Tabla de brechas respecto a Bayes
    gap_table = df.groupby(['Scenario', 'Method', 'n'])['Gap'].agg(['mean', 'std']).round(4)
    gap_table['Gap'] = gap_table.apply(lambda x: f"{x['mean']:+.4f} +- {x['std']:.4f}", axis=1)
    gap_pivot = gap_table['Gap'].unstack('n')
    
    print("\n\nBrecha respecto a Bayes (L(g) - L(Bayes)):")
    print(gap_pivot)
    
    return summary_pivot, gap_pivot



# Ejecutar análisis completo
if __name__ == "__main__":
    
    # Ejecutar simulación
    results_df = run_simulation()
    
    # Generar gráficas
    plot_results(results_df)
    
    # Generar tablas
    summary_table, gap_table = generate_summary_tables(results_df)
    
    
    # Guardar resultados
    #results_df.to_csv('resultados_completos_simulacion_nuevas.csv', index=False)
    #summary_table.to_csv('tabla_resumen_riesgos_nuevas.csv')
    #gap_table.to_csv('tabla_resumen_brechas_nuevas.csv')
