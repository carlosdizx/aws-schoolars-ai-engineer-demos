# Sistema RAG con Amazon Bedrock - Instrucciones de Uso

## Estado: IMPLEMENTADO Y FUNCIONANDO

El sistema RAG (Retrieval-Augmented Generation) ha sido implementado exitosamente y está listo para usar.

## Archivos Creados

1. **rag_system.py** - Sistema completo con comparaciones automáticas
2. **rag_interactive.py** - Versión interactiva con menú
3. **README_RAG.md** - Documentación completa
4. **requirements.txt** - Dependencias actualizadas

## Inicio Rápido

### 1. Instalar Dependencias (YA COMPLETADO)

```powershell
pip install chromadb sentence-transformers
```

**Estado:** ✓ Completado - Todas las dependencias instaladas

### 2. Verificar Credenciales AWS

Asegúrate de tener configuradas tus credenciales de AWS:

```powershell
aws configure
```

Necesitas:
- AWS Access Key ID
- AWS Secret Access Key
- Region: us-east-1 (o región con Bedrock habilitado)

### 3. Habilitar Modelos en AWS Bedrock

Ve a la consola de AWS Bedrock y habilita acceso a:
- **amazon.titan-embed-text-v1** (para embeddings)
- **anthropic.claude-3-haiku-20240307-v1:0** (para generación)

## Opciones de Ejecución

### Opción A: Sistema Completo con Comparaciones Automáticas

```powershell
python rag_system.py
```

**Qué hace:**
- Carga 10 documentos de ejemplo sobre AWS Bedrock y RAG
- Ejecuta 3 consultas de prueba
- Compara automáticamente respuestas CON RAG vs SIN RAG
- Muestra los resultados lado a lado

**Ideal para:** Ver cómo funciona RAG y sus beneficios

### Opción B: Sistema Interactivo

```powershell
python rag_interactive.py
```

**Qué hace:**
- Presenta un menú interactivo con 6 opciones:
  1. Hacer consulta con RAG
  2. Hacer consulta sin RAG
  3. Comparar RAG vs Sin RAG
  4. Agregar nuevos documentos
  5. Ver documentos actuales
  6. Salir

**Ideal para:** Experimentar con tus propias consultas y documentos

## Ejemplo de Uso Interactivo

```
Inicializando ChromaDB...
[OK] Coleccion de Chroma creada exitosamente

Cargando documentos de ejemplo...
[OK] 10 documentos cargados exitosamente

================================================================================
SISTEMA RAG INTERACTIVO - AMAZON BEDROCK
================================================================================

Opciones:
  1. Hacer una consulta con RAG
  2. Hacer una consulta sin RAG
  3. Comparar RAG vs Sin RAG
  4. Agregar nuevos documentos
  5. Ver documentos actuales
  6. Salir
================================================================================

Selecciona una opción (1-6): 3

================================================================================
COMPARACION: RAG vs SIN RAG
================================================================================

Ingresa tu consulta: ¿Qué es Amazon Bedrock?

Procesando con RAG...

Documentos recuperados:
  1. Amazon Bedrock es un servicio totalmente gestionado de modelos fundamentales.
  2. Amazon Bedrock proporciona acceso a modelos de IA de empresas líderes...
  3. Claude es un modelo de lenguaje desarrollado por Anthropic disponible...

Procesando sin RAG...

================================================================================
RESULTADOS DE LA COMPARACION
================================================================================

[RAG] CON RAG:
--------------------------------------------------------------------------------
Basado en el contexto proporcionado, Amazon Bedrock es un servicio totalmente
gestionado de modelos fundamentales que proporciona acceso a modelos de IA...
--------------------------------------------------------------------------------

[SIN RAG] SIN RAG:
--------------------------------------------------------------------------------
Amazon Bedrock es un servicio de AWS que permite...
--------------------------------------------------------------------------------
```

## Agregar Tus Propios Documentos

### Método 1: Usando el Sistema Interactivo

1. Ejecuta `python rag_interactive.py`
2. Selecciona opción 4 (Agregar nuevos documentos)
3. Ingresa tus documentos uno por línea
4. Escribe 'FIN' cuando termines

### Método 2: Modificando el Código

Edita `rag_system.py` o `rag_interactive.py` y modifica la lista `sample_docs`:

```python
sample_docs = [
    "Tu documento personalizado 1",
    "Tu documento personalizado 2",
    "Tu documento personalizado 3",
    # Agrega más documentos aquí
]
```

## Arquitectura del Sistema

```
Usuario → Query
    ↓
1. Embedding de Query (Titan Embed)
    ↓
2. Búsqueda en ChromaDB (Similitud Semántica)
    ↓
3. Recuperación de Top-K Documentos Relevantes
    ↓
4. Construcción de Prompt (Query + Contexto)
    ↓
5. Generación de Respuesta (Claude 3 Haiku)
    ↓
Respuesta Contextualizada
```

## Componentes Clave

### 1. BedrockEmbeddingFunction
- Convierte texto en vectores de alta dimensión
- Usa Amazon Titan Embed Text v1
- Compatible con ChromaDB

### 2. ChromaDB Collection
- Almacena embeddings de documentos
- Permite búsqueda por similitud semántica
- Recupera documentos relevantes eficientemente

### 3. Generación con Claude 3 Haiku
- Modelo rápido y económico
- Genera respuestas contextualizadas
- Soporta hasta 500 tokens de respuesta

## Parámetros Configurables

### En rag_system.py o rag_interactive.py:

```python
# Modelos
EMBEDDING_MODEL = "amazon.titan-embed-text-v1"
TEXT_GENERATION_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"

# Región
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

# Parámetros de generación
"max_tokens": 500,        # Longitud máxima de respuesta
"temperature": 0.7,       # 0.0-1.0 (creatividad)
"top_p": 0.9,            # 0.0-1.0 (diversidad)

# Número de documentos a recuperar
rag_generate(query, top_k=2)  # Cambia top_k según necesites
```

## Solución de Problemas

### Error: "Could not connect to Bedrock"
**Solución:**
- Verifica credenciales AWS: `aws configure`
- Confirma región correcta (us-east-1)
- Verifica acceso a Amazon Bedrock en tu cuenta

### Error: "Model access denied"
**Solución:**
1. Ve a AWS Console → Amazon Bedrock
2. Click en "Model access" en el menú lateral
3. Solicita acceso a:
   - Amazon Titan Embed Text v1
   - Anthropic Claude 3 Haiku
4. Espera aprobación (usualmente instantánea)

### Error: "ValidationException"
**Solución:**
- El modelo puede no estar disponible en tu región
- Cambia a us-east-1 o us-west-2
- Verifica que el modelo ID sea correcto

### Problemas de Encoding (caracteres raros)
**Solución:**
- Ya corregido en el código (sin emojis)
- Si persiste, ejecuta: `chcp 65001` en PowerShell

## Mejores Prácticas

### 1. Calidad de Documentos
- Usa documentos claros y concisos
- Evita duplicados
- Mantén documentos enfocados en un tema

### 2. Número de Documentos (top_k)
- **top_k=2**: Respuestas más enfocadas
- **top_k=3-5**: Balance entre contexto y precisión
- **top_k>5**: Más contexto pero puede diluir relevancia

### 3. Consultas Efectivas
- Sé específico en tus preguntas
- Usa términos relacionados con tus documentos
- Prueba diferentes formulaciones

### 4. Evaluación
- Compara siempre RAG vs Sin RAG
- Verifica que los documentos recuperados sean relevantes
- Ajusta top_k según resultados

## Casos de Uso Reales

1. **Chatbot Empresarial**
   - Carga manuales y políticas de la empresa
   - Responde preguntas de empleados con contexto preciso

2. **Asistente de Documentación Técnica**
   - Indexa documentación de productos
   - Responde preguntas técnicas con referencias exactas

3. **Sistema de Q&A sobre Productos**
   - Carga catálogos y especificaciones
   - Responde consultas de clientes con información actualizada

4. **Asistente Educativo**
   - Indexa material de curso
   - Responde preguntas de estudiantes basándose en el contenido

## Próximos Pasos

### Nivel Básico
- ✓ Ejecutar el sistema con documentos de ejemplo
- ✓ Probar consultas diferentes
- ✓ Comparar respuestas con y sin RAG

### Nivel Intermedio
- Agregar tus propios documentos
- Experimentar con diferentes valores de top_k
- Ajustar parámetros de generación (temperature, top_p)

### Nivel Avanzado
- Implementar persistencia de ChromaDB
- Agregar más documentos (100+)
- Crear interfaz web con Streamlit
- Implementar métricas de evaluación
- Agregar historial de conversación

## Recursos Adicionales

- **Documentación AWS Bedrock:** https://docs.aws.amazon.com/bedrock/
- **ChromaDB Docs:** https://docs.trychroma.com/
- **Anthropic Claude:** https://docs.anthropic.com/
- **Repositorio del Curso:** https://github.com/udacity/cd13926-Building-Apps-Amazon-Bedrock-exercises

## Contacto y Soporte

Si encuentras problemas:
1. Revisa esta guía de solución de problemas
2. Verifica los logs de error
3. Consulta la documentación de AWS Bedrock
4. Revisa el README_RAG.md para más detalles

---

**¡Sistema listo para usar! Ejecuta `python rag_system.py` o `python rag_interactive.py` para comenzar.**
