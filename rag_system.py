import boto3
import json
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

# Inicializar el cliente de Bedrock
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

# Configuración de modelos
EMBEDDING_MODEL = "amazon.titan-embed-text-v1"
TEXT_GENERATION_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"  # Modelo Claude 3 Haiku


class BedrockEmbeddingFunction(EmbeddingFunction):
    """
    Función de embedding personalizada para Amazon Bedrock
    Compatible con ChromaDB
    """
    
    def __init__(self):
        pass
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Obtiene embeddings para una lista de textos
        
        Args:
            input: Lista de textos a convertir en embeddings
            
        Returns:
            Lista de embeddings (cada uno es una lista de floats)
        """
        embeddings = []
        for text in input:
            body = json.dumps({"inputText": text})
            try:
                response = bedrock_runtime.invoke_model(
                    modelId=EMBEDDING_MODEL,
                    body=body,
                    contentType='application/json',
                    accept='application/json'
                )
                response_body = json.loads(response['body'].read())
                embeddings.append(response_body['embedding'])
            except Exception as e:
                print(f"Error obteniendo embedding: {e}")
                raise
        return embeddings


def generate_text(prompt):
    """
    Genera texto usando Claude 3 en Amazon Bedrock
    
    Args:
        prompt: El prompt para generar texto
        
    Returns:
        El texto generado por el modelo
    """
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "top_p": 0.9,
    })
    
    try:
        response = bedrock_runtime.invoke_model(
            modelId=TEXT_GENERATION_MODEL, 
            body=body,
            contentType='application/json',
            accept='application/json'
        )
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    except Exception as e:
        print(f"Error generando texto: {e}")
        raise


# Inicializar el cliente de Chroma
chroma_client = chromadb.Client()

# Crear una función de embedding personalizada que use Bedrock
bedrock_ef = BedrockEmbeddingFunction()

# Crear una colección con la función de embedding personalizada
try:
    # Intentar eliminar la colección si existe (para evitar errores en ejecuciones múltiples)
    try:
        chroma_client.delete_collection(name="bedrock_docs")
    except:
        pass
    
    collection = chroma_client.create_collection(
        name="bedrock_docs",
        embedding_function=bedrock_ef
    )
    print("[OK] Coleccion de Chroma creada exitosamente")
except Exception as e:
    print(f"Error creando colección: {e}")
    raise


def add_documents(docs):
    """
    Agrega documentos a la colección de Chroma
    
    Args:
        docs: Lista de documentos (strings) para indexar
    """
    try:
        collection.add(
            documents=docs,
            ids=[f"doc_{i}" for i in range(len(docs))]
        )
        print(f"[OK] {len(docs)} documentos agregados a la coleccion")
    except Exception as e:
        print(f"Error agregando documentos: {e}")
        raise


# Agregar documentos de ejemplo
sample_docs = [
    "Amazon Bedrock es un servicio totalmente gestionado de modelos fundamentales.",
    "Los sistemas RAG combinan recuperación y generación para mejorar las respuestas.",
    "Los embeddings son representaciones vectoriales de texto en espacios de alta dimensión.",
    "Chroma es un almacenamiento vectorial eficiente para construir aplicaciones de IA.",
    "Los modelos fundamentales pueden ajustarse para tareas y dominios específicos.",
    "Amazon Bedrock proporciona acceso a modelos de IA de empresas líderes como Anthropic, AI21 Labs y Amazon.",
    "RAG mejora la precisión de las respuestas al proporcionar contexto relevante del conocimiento almacenado.",
    "Los embeddings permiten buscar documentos similares mediante similitud de coseno.",
    "Claude es un modelo de lenguaje desarrollado por Anthropic disponible en Amazon Bedrock.",
    "Los sistemas RAG son especialmente útiles para aplicaciones que requieren conocimiento específico de dominio."
]

print("\nAgregando documentos de ejemplo...")
add_documents(sample_docs)


def rag_generate(query, top_k=2):
    """
    Genera una respuesta usando RAG (Retrieval-Augmented Generation)
    
    Args:
        query: La consulta del usuario
        top_k: Número de documentos relevantes a recuperar
        
    Returns:
        La respuesta generada con contexto
    """
    try:
        # Recuperar documentos relevantes
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Construir el prompt con el contexto recuperado
        context = "\n".join(results['documents'][0])
        
        prompt = f"""Dado el siguiente contexto, por favor responde la pregunta.

Contexto: {context}

Pregunta: {query}

Basado en el contexto proporcionado, mi respuesta es:"""
        
        # Generar respuesta
        response = generate_text(prompt)
        return response
    except Exception as e:
        print(f"Error en rag_generate: {e}")
        raise


def generate_without_rag(query):
    """
    Genera una respuesta sin usar RAG (sin contexto adicional)
    
    Args:
        query: La consulta del usuario
        
    Returns:
        La respuesta generada sin contexto
    """
    try:
        prompt = query
        return generate_text(prompt)
    except Exception as e:
        print(f"Error en generate_without_rag: {e}")
        raise


def main():
    """Función principal para probar el sistema RAG"""
    print("\n" + "="*80)
    print("SISTEMA RAG CON AMAZON BEDROCK")
    print("="*80 + "\n")
    
    # Probar el sistema RAG con una consulta
    print("Prueba inicial del sistema RAG:\n")
    query = "¿Cómo se relaciona Amazon Bedrock con los sistemas RAG?"
    print(f"Consulta: {query}\n")
    
    response = rag_generate(query)
    print(f"Respuesta RAG: {response}\n")
    
    # Comparar respuestas con y sin RAG
    print("\n" + "="*80)
    print("COMPARACION: RAG vs Sin RAG")
    print("="*80 + "\n")
    
    test_queries = [
        "¿Para qué se utilizan los embeddings en IA?",
        "Explica los beneficios de usar RAG en aplicaciones de IA.",
        "¿Cómo soporta Amazon Bedrock los modelos fundamentales?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Consulta: {query}")
        print(f"{'='*80}\n")
        
        print("[RAG] Respuesta con RAG:")
        print("-" * 80)
        rag_response = rag_generate(query)
        print(rag_response)
        
        print("\n\n[SIN RAG] Respuesta sin RAG:")
        print("-" * 80)
        no_rag_response = generate_without_rag(query)
        print(no_rag_response)
        
        print("\n" + "="*80)


if __name__ == "__main__":
    main()
