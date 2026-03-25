import streamlit as st
import tempfile
import os
import subprocess
import torch
import nemo.collections.asr as nemo_asr

st.set_page_config(page_title="Canary-1B-v2 ASR & Traducción", layout="wide")

# Inicializar y cachear el modelo de NeMo
@st.cache_resource
def load_model():
    with st.spinner("Cargando modelo Canary-1B-v2... Esto puede tomar un momento."):
        # Cargar modelo base
        model = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-1b-v2")
        # Optimizando memoria pasándolo a 16-bits si es posible
        if torch.cuda.is_available():
            model = model.bfloat16()
            model = model.to('cuda')
        return model

# Título de la app
st.title("🪶 Demostración Interactiva: Canary-1B-v2")
st.markdown("""
Esta aplicación permite cargar un archivo de audio y utilizar el modelo **Canary-1B-v2** de NVIDIA NeMo para tareas de Reconocimiento Automático del Habla (ASR) o Traducción (S2T).
""")

# Barra Lateral de Configuración
st.sidebar.header("⚙️ Configuración de Inferencia")
task = st.sidebar.selectbox(
    "Tarea", 
    options=["asr", "s2t_translation"], 
    index=0, 
    help="ASR para transcribir en el mismo idioma, s2t_translation para traducir a otro idioma."
)

source_lang = st.sidebar.selectbox(
    "Idioma de Origen del Audio", 
    options=["es", "en", "fr", "de"], 
    index=0
)

target_lang_options = ["es", "en", "fr", "de"]

target_lang = st.sidebar.selectbox("Idioma de Destino", options=target_lang_options, index=1)

pnc = st.sidebar.radio(
    "Incluir Puntuación y Mayúsculas (PNC)", 
    options=["yes", "no"], 
    index=0
)

# Cargar Modelo
model = load_model()
st.sidebar.success("✅ Modelo cargado y listo en memoria.")

# Sección principal
audio_file = st.file_uploader("📂 Sube tu archivo de audio (.wav, .mp3, .flac, .m4a)", type=["wav", "mp3", "flac", "m4a"])

if audio_file is not None:
    # Mostrar reproductor de audio
    st.audio(audio_file, format=f"audio/{audio_file.name.split('.')[-1]}")
    
    if st.button("🚀 Iniciar Inferencia", use_container_width=True):
        with st.spinner("Procesando el audio con Canary..."):
            # Guardar el archivo subido en temporal para procesarlo con NeMo
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(audio_file.getvalue())
                tmp_audio_path = tmp_file.name
            
            mono_audio_path = tmp_audio_path + "_mono.wav"
            
            try:
                # Convertir a mono (1 canal) usando ffmpeg
                subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp_audio_path, "-ac", "1", mono_audio_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )

                # Ejecutando la inferencia de Canary
                transcription = model.transcribe(
                    audio=[mono_audio_path],
                    batch_size=1,
                    task=task,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    pnc=pnc
                )
                
                # Formatear el resultado
                result = transcription[0] if isinstance(transcription, list) else transcription
                
                st.success("¡Inferencia completada con éxito!")
                
                # Preparar HTML personalizado asegurando 0 indentación en el string
                # para que el parser de Markdown de Streamlit no lo convierta en bloque de código
                html_parts = []
                html_parts.append('<div style="background-color: #1e1e2e; padding: 20px; border-radius: 10px; border: 1px solid #313244; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">')
                html_parts.append('<h3 style="color: #cba6f7; margin-top: 0; margin-bottom: 15px;">🎯 Resultado de la Inferencia</h3>')
                
                if hasattr(result, 'text'):
                    html_parts.append('<p style="color: #a6e3a1; font-weight: bold; margin-bottom: 8px;">🗣️ Texto Transcrito:</p>')
                    html_parts.append(f'<div style="background-color: #181825; padding: 12px; border-radius: 6px; color: #cdd6f4; font-size: 16px; margin-bottom: 20px; border-left: 4px solid #a6e3a1;">{result.text}</div>')
                    
                if hasattr(result, 'y_sequence'):
                    tokens = result.y_sequence.tolist()
                    html_parts.append('<p style="color: #fab387; font-weight: bold; margin-bottom: 8px;">🔢 Secuencia de Tokens:</p>')
                    html_parts.append(f'<div style="background-color: #181825; padding: 12px; border-radius: 6px; color: #f38ba8; font-family: monospace; font-size: 14px; word-wrap: break-word; border-left: 4px solid #fab387;">{tokens}</div>')
                
                html_parts.append('</div>')
                
                # Mostrar el contenedor unificado permitiendo HTML
                st.markdown("".join(html_parts), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error durante la inferencia: {str(e)}")
            finally:
                # Limpiar el archivo temporal
                if os.path.exists(tmp_audio_path):
                    os.unlink(tmp_audio_path)
                if 'mono_audio_path' in locals() and os.path.exists(mono_audio_path):
                    os.unlink(mono_audio_path)

