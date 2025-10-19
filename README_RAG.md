# Sistema RAG con Amazon Bedrock

Este proyecto implementa un sistema de **Generación Aumentada por Recuperación (RAG)** usando Amazon Bedrock, ChromaDB y modelos de IA de última generación.

## 🎯 Objetivos del Ejercicio

- Implementar un sistema RAG básico usando Amazon Bedrock
- Seleccionar modelos adecuados para embeddings y generación de texto
- Desarrollar indexación de documentos con embeddings
- Implementar recuperación basada en similitud semántica
- Comparar respuestas con y sin RAG

## 📋 Requisitos Previos

- Cuenta de AWS con acceso a Amazon Bedrock
- Python 3.8 o superior
- Credenciales de AWS configuradas
- Acceso habilitado a los modelos:
  - `amazon.titan-embed-text-v1` (para embeddings)
  - `anthropic.claude-3-haiku-20240307-v1:0` (para generación de texto)

## 🚀 Configuración del Entorno

### 1. Crear y activar entorno virtual

**En Windows:**
```bash
python -m venv rag-env
rag-env\Scripts\activate
```

**En Linux/Mac:**
```bash
python -m venv rag-env
source rag-env/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar credenciales de AWS

Si aún no has configurado tus credenciales de AWS:

```bash
aws configure
```

Ingresa:
- AWS Access Key ID
- AWS Secret Access Key
- Default region name: `us-east-1`
- Default output format: `json`

## 🏗️ Arquitectura del Sistema

```
┌─────────────┐
│   Usuario   │
└──────┬──────┘
       │ Query
       ▼
┌─────────────────────────────────────┐
│         Sistema RAG                 │
│                                     │
│  1. Embedding de Query              │
│     (Titan Embed)                   │
│          │                          │
│          ▼                          │
│  2. Búsqueda en ChromaDB            │
│     (Similitud Semántica)           │
│          │                          │
│          ▼                          │
│  3. Recuperación de Documentos      │
│          │                          │
│          ▼                          │
│  4. Construcción de Prompt          │
│     (Query + Contexto)              │
│          │                          │
│          ▼                          │
│  5. Generación de Respuesta         │
│     (Claude 3 Haiku)                │
└─────────────────────────────────────┘
```

## 📁 Estructura del Proyecto

```
aws-schoolars-ai-engineer-demos/
│
├── rag_system.py          # Sistema RAG completo
├── requirements.txt       # Dependencias del proyecto
├── README_RAG.md         # Este archivo
├── main.py               # Demo de conversación con Bedrock
└── example.py            # Ejemplo de Knowledge Base
```

## 🔧 Componentes del Sistema

### 1. **Modelo de Embeddings**
- **Modelo:** `amazon.titan-embed-text-v1`
- **Función:** Convierte texto en vectores de alta dimensión
- **Uso:** Indexación de documentos y búsqueda semántica

### 2. **Base de Datos Vectorial**
- **Tecnología:** ChromaDB
- **Función:** Almacena y busca embeddings eficientemente
- **Características:** Búsqueda por similitud de coseno

### 3. **Modelo de Generación**
- **Modelo:** `anthropic.claude-3-haiku-20240307-v1:0`
- **Función:** Genera respuestas contextualizadas
- **Ventajas:** Rápido, preciso y cost-effective

## 💻 Uso del Sistema

### Ejecutar el sistema completo

```bash
python rag_system.py
```

Este script:
1. ✅ Inicializa la conexión con Amazon Bedrock
2. ✅ Crea una colección en ChromaDB
3. ✅ Indexa documentos de ejemplo
4. ✅ Realiza consultas de prueba
5. ✅ Compara respuestas con y sin RAG

### Funciones Principales

#### `get_bedrock_embedding(text)`
Obtiene el embedding de un texto usando Amazon Titan.

```python
embedding = get_bedrock_embedding("Amazon Bedrock es increíble")
```

#### `generate_text(prompt)`
Genera texto usando Claude 3 Haiku.

```python
response = generate_text("¿Qué es RAG?")
```

#### `add_documents(docs)`
Agrega documentos a la base de datos vectorial.

```python
docs = ["Documento 1", "Documento 2"]
add_documents(docs)
```

#### `rag_generate(query, top_k=2)`
Genera respuesta usando RAG (con contexto).

```python
response = rag_generate("¿Qué es Amazon Bedrock?", top_k=2)
```

#### `generate_without_rag(query)`
Genera respuesta sin RAG (sin contexto adicional).

```python
response = generate_without_rag("¿Qué es Amazon Bedrock?")
```

## 🧪 Ejemplo de Uso Personalizado

```python
import boto3
from rag_system import rag_generate, add_documents

# Agregar tus propios documentos
mis_documentos = [
    "Tu documento personalizado 1",
    "Tu documento personalizado 2",
    "Tu documento personalizado 3"
]
add_documents(mis_documentos)

# Hacer consultas
query = "¿Qué información tienes sobre...?"
respuesta = rag_generate(query, top_k=3)
print(respuesta)
```

## 📊 Comparación: RAG vs Sin RAG

El sistema automáticamente compara respuestas:

| Aspecto | Con RAG | Sin RAG |
|---------|---------|---------|
| **Precisión** | Alta (usa contexto específico) | Variable |
| **Relevancia** | Alta (documentos relevantes) | Depende del conocimiento del modelo |
| **Actualización** | Fácil (agregar documentos) | Requiere reentrenamiento |
| **Costo** | Moderado | Bajo |

## 🔍 Casos de Uso

1. **Chatbots empresariales** - Respuestas basadas en documentación interna
2. **Asistentes de soporte** - Búsqueda en base de conocimiento
3. **Análisis de documentos** - Extracción de información relevante
4. **Q&A sobre productos** - Respuestas precisas sobre catálogos
5. **Asistentes educativos** - Respuestas basadas en material de curso

## ⚙️ Configuración Avanzada

### Cambiar el modelo de generación

Edita `TEXT_GENERATION_MODEL` en `rag_system.py`:

```python
# Opciones disponibles:
TEXT_GENERATION_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"  # Rápido y económico
TEXT_GENERATION_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"  # Balance
TEXT_GENERATION_MODEL = "anthropic.claude-v2:1"  # Claude v2
```

### Ajustar parámetros de generación

```python
body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1000,        # Aumentar para respuestas más largas
    "temperature": 0.5,        # 0.0-1.0 (menor = más determinista)
    "top_p": 0.9,             # 0.0-1.0 (control de diversidad)
    "messages": [...]
})
```

### Cambiar número de documentos recuperados

```python
# Recuperar más documentos para mayor contexto
response = rag_generate(query, top_k=5)
```

## 🐛 Solución de Problemas

### Error: "Could not connect to Bedrock"
- Verifica tus credenciales de AWS
- Confirma que tienes acceso a Amazon Bedrock
- Verifica la región (debe ser `us-east-1` o región con Bedrock habilitado)

### Error: "Model access denied"
- Ve a la consola de AWS Bedrock
- Solicita acceso a los modelos necesarios
- Espera la aprobación (puede tomar unos minutos)

### Error: "ChromaDB collection already exists"
- El script automáticamente elimina colecciones existentes
- Si persiste, reinicia el script

## 📚 Recursos Adicionales

- [Documentación de Amazon Bedrock](https://docs.aws.amazon.com/bedrock/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)
- [Repositorio del curso](https://github.com/udacity/cd13926-Building-Apps-Amazon-Bedrock-exercises)

## 🎓 Aprendizajes Clave

1. **RAG mejora significativamente la precisión** de las respuestas al proporcionar contexto relevante
2. **Los embeddings permiten búsqueda semántica** más allá de coincidencias de palabras clave
3. **ChromaDB facilita la gestión** de bases de datos vectoriales
4. **Amazon Bedrock simplifica** el acceso a modelos de IA de última generación
5. **La combinación de recuperación y generación** crea sistemas más robustos

## 📝 Notas

- Los embeddings se generan automáticamente al agregar documentos
- La búsqueda usa similitud de coseno por defecto
- El sistema es stateless (no mantiene historial de conversación)
- Para producción, considera usar bases de datos vectoriales persistentes

## 🚀 Próximos Pasos

1. Agregar más documentos específicos de tu dominio
2. Implementar persistencia de la base de datos vectorial
3. Agregar manejo de historial de conversación
4. Implementar métricas de evaluación
5. Crear una interfaz web con Streamlit o Gradio

---

**¡Feliz aprendizaje con RAG y Amazon Bedrock!** 🎉
