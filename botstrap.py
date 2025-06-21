import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="Bootstrap Estadístico",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar el diseño
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .case-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem;
    }
    
    .results-section {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .interpretation-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown("""
<div class="main-header">
    <h1>🎯 Aplicación Bootstrap Estadístico</h1>
    <p>Herramienta interactiva para análisis Bootstrap con 5 casos de estudio</p>
</div>
""", unsafe_allow_html=True)

# Sidebar para selección
st.sidebar.title("🎛️ Panel de Control")
st.sidebar.markdown("---")

casos = {
    "Caso 1: Media - Alturas de Estudiantes": "media",
    "Caso 2: Mediana - Salarios": "mediana", 
    "Caso 3: Desviación Estándar - Tiempos de Producción": "std",
    "Caso 4: Proporción - Control de Calidad": "proporcion",
    "Caso 5: Percentil 90 - Puntuaciones": "percentil90"
}

caso_seleccionado = st.sidebar.selectbox(
    "🎯 Selecciona un caso de estudio:",
    list(casos.keys())
)

# Parámetros del Bootstrap
st.sidebar.markdown("### ⚙️ Configuración Bootstrap")
num_bootstrap = st.sidebar.slider("Número de iteraciones Bootstrap", 100, 5000, 1000)
mostrar_detalles = st.sidebar.checkbox("Mostrar detalles del proceso", value=False)
nivel_confianza = st.sidebar.slider("Nivel de confianza (%)", 90, 99, 95)

# Opción para datos personalizados
st.sidebar.markdown("### 📝 Datos Personalizados")
usar_datos_personalizados = st.sidebar.checkbox("Usar mis propios datos", value=False)

# Funciones para cada estadístico
def calculate_media(sample):
    return np.mean(sample)

def calculate_mediana(sample):
    return np.median(sample)

def calculate_std(sample):
    return np.std(sample, ddof=1)

def calculate_proporcion(sample):
    return np.mean(sample)

def calculate_percentil90(sample):
    return np.percentile(sample, 90)

# Función principal de Bootstrap
def bootstrap_process(data, statistic_func, B=1000, show_details=False):
    n = len(data)
    bootstrap_stats = []
    
    if show_details:
        st.write(f"🔄 Iniciando Bootstrap con {B} iteraciones...")
        progress_bar = st.progress(0)
    
    for i in range(B):
        indices = np.random.choice(n, size=n, replace=True)
        bootstrap_sample = data[indices]
        bootstrap_stat = statistic_func(bootstrap_sample)
        bootstrap_stats.append(bootstrap_stat)
        
        if show_details and i < 5:
            st.write(f"**Muestra #{i+1}:** {bootstrap_sample}")
            st.write(f"   → **Estadístico:** {bootstrap_stat:.4f}")
        
        if show_details and i % 100 == 0:
            progress_bar.progress((i + 1) / B)
    
    return np.array(bootstrap_stats)

# Función para crear visualizaciones
def create_plots(bootstrap_results, original_stat, stat_name, data):
    # Crear subplots con Plotly
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(f'Distribución Bootstrap del {stat_name}', 
                       f'Datos Originales',
                       f'Diagrama de Caja - Bootstrap',
                       f'Q-Q Plot'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Histograma Bootstrap
    fig.add_trace(
        go.Histogram(x=bootstrap_results, nbinsx=30, name='Bootstrap', 
                    marker_color='rgba(55, 128, 191, 0.7)',
                    histnorm='probability density'),
        row=1, col=1
    )
    
    # Líneas verticales para estadísticos
    fig.add_vline(x=original_stat, line_dash="dash", line_color="red", 
                  annotation_text=f"{stat_name} Original", row=1, col=1)
    fig.add_vline(x=np.mean(bootstrap_results), line_dash="dash", line_color="green",
                  annotation_text="Media Bootstrap", row=1, col=1)
    
    # Datos originales
    fig.add_trace(
        go.Bar(x=list(range(len(data))), y=data, name='Datos Originales',
               marker_color='rgba(255, 153, 51, 0.8)'),
        row=1, col=2
    )
    
    # Box plot
    fig.add_trace(
        go.Box(y=bootstrap_results, name='Bootstrap Results',
               marker_color='rgba(50, 171, 96, 0.8)'),
        row=2, col=1
    )
    
    # Q-Q Plot aproximado
    sorted_bootstrap = np.sort(bootstrap_results)
    theoretical_quantiles = np.linspace(0, 1, len(sorted_bootstrap))
    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sorted_bootstrap, 
                  mode='markers', name='Q-Q Plot',
                  marker=dict(color='purple', size=4)),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, 
                     title_text=f"Análisis Bootstrap Completo - {stat_name}")
    
    return fig

# Datos para cada caso
datos_casos = {
    "media": {
        "data": np.array([165, 170, 175, 168, 172, 169, 173, 167, 171, 174]),
        "func": calculate_media,
        "name": "Media",
        "description": "Alturas de estudiantes (cm)",
        "context": "Un profesor tiene las alturas de 10 estudiantes y quiere estimar la altura promedio de TODA la población estudiantil."
    },
    "mediana": {
        "data": np.array([25000, 30000, 32000, 28000, 35000, 31000, 85000, 29000, 33000, 30000]),
        "func": calculate_mediana,
        "name": "Mediana",
        "description": "Salarios de empleados ($)",
        "context": "Una empresa quiere estimar el salario mediano de su sector basándose en una muestra de 10 empleados."
    },
    "std": {
        "data": np.array([2.1, 2.5, 2.3, 2.8, 2.4, 2.6, 2.2, 2.7, 2.3, 2.5]),
        "func": calculate_std,
        "name": "Desviación Estándar",
        "description": "Tiempos de producción (horas)",
        "context": "Una fábrica mide los tiempos de producción de 10 piezas y quiere estimar la variabilidad del proceso completo."
    },
    "proporcion": {
        "data": np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 0]),
        "func": calculate_proporcion,
        "name": "Proporción",
        "description": "Control de calidad (1=Pasa, 0=Falla)",
        "context": "Una empresa de manufactura inspecciona productos y quiere estimar la proporción de productos que pasan el control de calidad."
    },
    "percentil90": {
        "data": np.array([78, 85, 92, 88, 91, 87, 89, 93, 86, 90]),
        "func": calculate_percentil90,
        "name": "Percentil 90",
        "description": "Puntuaciones de examen",
        "context": "Un profesor registra las puntuaciones de sus estudiantes y quiere estimar el percentil 90 de toda la población estudiantil."
    }
}

# Procesar caso seleccionado
caso_key = casos[caso_seleccionado]
caso_info = datos_casos[caso_key].copy()

# Permitir entrada de datos personalizados
if usar_datos_personalizados:
    st.sidebar.markdown("#### ✏️ Ingresa tus datos")
    
    if caso_key == "proporcion":
        st.sidebar.info("Para proporciones, usa 1 para éxito y 0 para fallo")
        datos_texto = st.sidebar.text_area(
            "Datos (separados por comas):",
            value="1,1,0,1,1,1,0,1,1,0",
            help="Ejemplo: 1,1,0,1,1,1,0,1,1,0"
        )
    else:
        # Mostrar datos por defecto como ejemplo
        datos_default = ",".join(map(str, caso_info['data']))
        datos_texto = st.sidebar.text_area(
            "Datos (separados por comas):",
            value=datos_default,
            help="Ejemplo: 165,170,175,168,172"
        )
    
    try:
        # Convertir texto a array
        datos_personalizados = np.array([float(x.strip()) for x in datos_texto.split(',')])
        
        # Validar datos para proporciones
        if caso_key == "proporcion":
            if not all(x in [0, 1] for x in datos_personalizados):
                st.sidebar.error("⚠️ Para proporciones, usa solo 0 y 1")
                datos_personalizados = caso_info['data']
        
        # Actualizar los datos del caso
        caso_info['data'] = datos_personalizados
        
        st.sidebar.success(f"✅ {len(datos_personalizados)} datos cargados correctamente")
        
        # Mostrar estadísticas básicas de los datos personalizados
        st.sidebar.markdown("**📊 Estadísticas básicas:**")
        st.sidebar.write(f"• Mínimo: {np.min(datos_personalizados):.2f}")
        st.sidebar.write(f"• Máximo: {np.max(datos_personalizados):.2f}")
        st.sidebar.write(f"• Media: {np.mean(datos_personalizados):.2f}")
        
    except ValueError:
        st.sidebar.error("❌ Error: Verifica que todos los valores sean números separados por comas")
        # Mantener datos originales si hay error

# Mostrar información del caso
data_source = "personalizados" if usar_datos_personalizados else "predeterminados"
st.markdown(f"""
<div class="case-card">
    <h2>📋 {caso_seleccionado}</h2>
    <p><strong>Descripción:</strong> {caso_info['description']}</p>
    <p><strong>Contexto:</strong> {caso_info['context']}</p>
    <p><strong>Datos:</strong> Usando datos {data_source} ({len(caso_info['data'])} valores)</p>
</div>
""", unsafe_allow_html=True)

# Mostrar datos originales
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Datos Originales")
    if usar_datos_personalizados:
        st.success("✨ Usando tus datos personalizados")
    else:
        st.info("📋 Usando datos de ejemplo predeterminados")
    
    st.write(f"**Datos ({len(caso_info['data'])} valores):** {caso_info['data']}")
    
    # Crear tabla de datos
    df_data = pd.DataFrame({
        'Índice': range(1, len(caso_info['data']) + 1),
        'Valor': caso_info['data']
    })
    st.dataframe(df_data, use_container_width=True)
    
    # Estadísticas descriptivas básicas
    st.markdown("**📈 Estadísticas Descriptivas:**")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Media", f"{np.mean(caso_info['data']):.3f}")
    with col_b:
        st.metric("Mediana", f"{np.median(caso_info['data']):.3f}")
    with col_c:
        st.metric("Desv. Est.", f"{np.std(caso_info['data'], ddof=1):.3f}")

with col2:
    st.subheader("📈 Visualización de Datos")
    fig_data = px.bar(x=range(1, len(caso_info['data']) + 1), 
                      y=caso_info['data'],
                      title=f"Datos Originales - {caso_info['description']}")
    fig_data.update_traces(marker_color='rgba(255, 153, 51, 0.8)')
    st.plotly_chart(fig_data, use_container_width=True)

# Calcular estadístico original
original_stat = caso_info['func'](caso_info['data'])

# Botón para ejecutar Bootstrap
if st.button("🚀 Ejecutar Análisis Bootstrap", type="primary"):
    with st.spinner('Ejecutando Bootstrap...'):
        # Ejecutar Bootstrap
        bootstrap_results = bootstrap_process(
            caso_info['data'], 
            caso_info['func'], 
            B=num_bootstrap, 
            show_details=mostrar_detalles
        )
        
        st.success("✅ ¡Análisis Bootstrap completado!")
        
        # Mostrar resultados principales
        st.markdown("""
        <div class="results-section">
            <h3>🎯 Resultados del Análisis Bootstrap</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>{caso_info['name']} Original</h4>
                <h2>{original_stat:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Media Bootstrap</h4>
                <h2>{np.mean(bootstrap_results):.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Error Estándar</h4>
                <h2>{np.std(bootstrap_results, ddof=1):.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        alpha = (100 - nivel_confianza) / 100
        ci_lower = np.percentile(bootstrap_results, (alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_results, (1 - alpha/2) * 100)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>IC {nivel_confianza}%</h4>
                <h2>[{ci_lower:.3f}, {ci_upper:.3f}]</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualizaciones
        st.subheader("📊 Visualizaciones del Análisis")
        fig = create_plots(bootstrap_results, original_stat, caso_info['name'], caso_info['data'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas adicionales
        st.subheader("📋 Estadísticas Detalladas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            stats_df = pd.DataFrame({
                'Estadística': [
                    f'{caso_info["name"]} Original',
                    'Media Bootstrap',
                    'Mediana Bootstrap', 
                    'Error Estándar Bootstrap',
                    'Varianza Bootstrap',
                    'Mínimo Bootstrap',
                    'Máximo Bootstrap'
                ],
                'Valor': [
                    f'{original_stat:.6f}',
                    f'{np.mean(bootstrap_results):.6f}',
                    f'{np.median(bootstrap_results):.6f}',
                    f'{np.std(bootstrap_results, ddof=1):.6f}',
                    f'{np.var(bootstrap_results, ddof=1):.6f}',
                    f'{np.min(bootstrap_results):.6f}',
                    f'{np.max(bootstrap_results):.6f}'
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            # Percentiles
            percentiles = [5, 10, 25, 50, 75, 90, 95]
            percentile_values = [np.percentile(bootstrap_results, p) for p in percentiles]
            
            perc_df = pd.DataFrame({
                'Percentil': [f'P{p}' for p in percentiles],
                'Valor': [f'{v:.6f}' for v in percentile_values]
            })
            st.dataframe(perc_df, use_container_width=True)
        
        # Interpretación
        st.markdown(f"""
        <div class="interpretation-box">
            <h3>🎓 Interpretación de Resultados</h3>
            <ol>
                <li><strong>Distribución Bootstrap:</strong> El Bootstrap nos permitió estimar la distribución del {caso_info['name']} sin asumir normalidad de los datos.</li>
                <li><strong>Error Estándar:</strong> El error estándar bootstrap ({np.std(bootstrap_results, ddof=1):.4f}) mide la variabilidad de nuestra estimación del {caso_info['name']}.</li>
                <li><strong>Intervalo de Confianza:</strong> Con {nivel_confianza}% de confianza, el verdadero {caso_info['name']} poblacional está entre {ci_lower:.4f} y {ci_upper:.4f}.</li>
                <li><strong>Sesgo Bootstrap:</strong> El sesgo estimado es {np.mean(bootstrap_results) - original_stat:.6f}, lo que indica {'sobrestimación' if np.mean(bootstrap_results) > original_stat else 'subestimación' if np.mean(bootstrap_results) < original_stat else 'estimación no sesgada'}.</li>
                <li><strong>Distribución Muestral:</strong> La distribución bootstrap aproxima la verdadera distribución muestral del estadístico.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Opción para descargar resultados
        results_dict = {
            'case': caso_seleccionado,
            'data_used': caso_info['data'].tolist(),
            'custom_data': usar_datos_personalizados,
            'bootstrap_results': bootstrap_results.tolist(),
            'original_statistic': float(original_stat),
            'bootstrap_mean': float(np.mean(bootstrap_results)),
            'bootstrap_std': float(np.std(bootstrap_results, ddof=1)),
            'confidence_interval': [float(ci_lower), float(ci_upper)],
            'num_iterations': num_bootstrap,
            'confidence_level': nivel_confianza,
            'bias': float(np.mean(bootstrap_results) - original_stat)
        }
        
        st.download_button(
            label="📥 Descargar Resultados (JSON)",
            data=pd.Series(results_dict).to_json(),
            file_name=f"bootstrap_results_{caso_key}.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎯 <strong>Aplicación Bootstrap Estadístico</strong> | Desarrollado para análisis estadístico avanzado</p>
    <p>📊 Herramienta educativa para comprender el método Bootstrap</p>
</div>
""", unsafe_allow_html=True)