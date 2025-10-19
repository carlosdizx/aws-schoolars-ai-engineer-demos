"""
Sistema RAG Interactivo con Amazon Bedrock
Permite hacer consultas interactivas y comparar respuestas con y sin RAG
"""

import boto3
import json
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

# Inicializar el cliente de Bedrock
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

# Configuración de modelos
EMBEDDING_MODEL = "amazon.titan-embed-text-v1"
TEXT_GENERATION_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"


class BedrockEmbeddingFunction(EmbeddingFunction):
    """Función de embedding personalizada para Amazon Bedrock"""
    
    def __init__(self):
        pass
    
    def __call__(self, input: Documents) -> Embeddings:
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
                print(f"[ERROR] Error obteniendo embedding: {e}")
                raise
        return embeddings


def generate_text(prompt):
    """Genera texto usando Claude 3 en Amazon Bedrock"""
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
        print(f"[ERROR] Error generando texto: {e}")
        raise


# Inicializar ChromaDB
print("Inicializando ChromaDB...")
chroma_client = chromadb.Client()

# Crear función de embedding personalizada
bedrock_ef = BedrockEmbeddingFunction()

# Crear colección
try:
    try:
        chroma_client.delete_collection(name="bedrock_docs")
    except:
        pass
    
    collection = chroma_client.create_collection(
        name="bedrock_docs",
        embedding_function=bedrock_ef
    )
    print("[OK] Coleccion de Chroma creada exitosamente\n")
except Exception as e:
    print(f"[ERROR] Error creando coleccion: {e}")
    exit(1)


def add_documents(docs):
    """Agrega documentos a la colección de Chroma"""
    try:
        collection.add(
            documents=docs,
            ids=[f"doc_{i}" for i in range(len(docs))]
        )
        return True
    except Exception as e:
        print(f"[ERROR] Error agregando documentos: {e}")
        return False


def rag_generate(query, top_k=2, verbose=False):
    """Genera una respuesta usando RAG"""
    try:
        # Recuperar documentos relevantes
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Mostrar documentos recuperados si verbose está activado
        if verbose:
            print("\nDocumentos recuperados:")
            for i, doc in enumerate(results['documents'][0], 1):
                print(f"  {i}. {doc}")
            print()
        
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
        print(f"[ERROR] Error en rag_generate: {e}")
        return None


def generate_without_rag(query):
    """Genera una respuesta sin usar RAG"""
    try:
        return generate_text(query)
    except Exception as e:
        print(f"[ERROR] Error en generate_without_rag: {e}")
        return None


def show_menu():
    """Muestra el menú principal"""
    print("\n" + "="*80)
    print("SISTEMA RAG INTERACTIVO - AMAZON BEDROCK")
    print("="*80)
    print("\nOpciones:")
    print("  1. Hacer una consulta con RAG")
    print("  2. Hacer una consulta sin RAG")
    print("  3. Comparar RAG vs Sin RAG")
    print("  4. Agregar nuevos documentos")
    print("  5. Ver documentos actuales")
    print("  6. Salir")
    print("="*80)


def view_documents():
    """Muestra todos los documentos en la colección"""
    try:
        # Obtener todos los documentos
        results = collection.get()
        docs = results.get('documents', [])
        
        if not docs:
            print("\nNo hay documentos en la coleccion.")
            return
        
        print(f"\nDocumentos en la coleccion ({len(docs)} total):")
        print("-"*80)
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc}")
        print("-"*80)
    except Exception as e:
        print(f"[ERROR] Error obteniendo documentos: {e}")


def main():
    """Función principal del sistema interactivo"""
    
    # Agregar documentos de ejemplo iniciales
    print("Cargando documentos de ejemplo...")
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
    
    if add_documents(sample_docs):
        print(f"[OK] {len(sample_docs)} documentos cargados exitosamente\n")
    else:
        print("[ERROR] Error cargando documentos iniciales")
        return
    
    # Loop principal
    while True:
        show_menu()
        choice = input("\nSelecciona una opción (1-6): ").strip()
        
        if choice == '1':
            # Consulta con RAG
            print("\n" + "="*80)
            print("CONSULTA CON RAG")
            print("="*80)
            query = input("\nIngresa tu consulta: ").strip()
            
            if query:
                print("\nProcesando con RAG...")
                response = rag_generate(query, top_k=3, verbose=True)
                if response:
                    print("Respuesta:")
                    print("-"*80)
                    print(response)
                    print("-"*80)
        
        elif choice == '2':
            # Consulta sin RAG
            print("\n" + "="*80)
            print("CONSULTA SIN RAG")
            print("="*80)
            query = input("\nIngresa tu consulta: ").strip()
            
            if query:
                print("\nProcesando sin RAG...")
                response = generate_without_rag(query)
                if response:
                    print("Respuesta:")
                    print("-"*80)
                    print(response)
                    print("-"*80)
        
        elif choice == '3':
            # Comparar RAG vs Sin RAG
            print("\n" + "="*80)
            print("COMPARACION: RAG vs SIN RAG")
            print("="*80)
            query = input("\nIngresa tu consulta: ").strip()
            
            if query:
                print("\nProcesando con RAG...")
                rag_response = rag_generate(query, top_k=3, verbose=True)
                
                print("\nProcesando sin RAG...")
                no_rag_response = generate_without_rag(query)
                
                print("\n" + "="*80)
                print("RESULTADOS DE LA COMPARACION")
                print("="*80)
                
                print("\n[RAG] CON RAG:")
                print("-"*80)
                if rag_response:
                    print(rag_response)
                print("-"*80)
                
                print("\n[SIN RAG] SIN RAG:")
                print("-"*80)
                if no_rag_response:
                    print(no_rag_response)
                print("-"*80)
        
        elif choice == '4':
            # Agregar nuevos documentos
            print("\n" + "="*80)
            print("AGREGAR NUEVOS DOCUMENTOS")
            print("="*80)
            print("\nIngresa los documentos (uno por línea).")
            print("Escribe 'FIN' cuando termines:\n")
            
            new_docs = []
            while True:
                doc = input(f"Documento {len(new_docs) + 1}: ").strip()
                if doc.upper() == 'FIN':
                    break
                if doc:
                    new_docs.append(doc)
            
            if new_docs:
                print(f"\nAgregando {len(new_docs)} documentos...")
                if add_documents(new_docs):
                    print(f"[OK] {len(new_docs)} documentos agregados exitosamente")
            else:
                print("[AVISO] No se agregaron documentos")
        
        elif choice == '5':
            # Ver documentos actuales
            view_documents()
        
        elif choice == '6':
            # Salir
            print("\nGracias por usar el Sistema RAG!")
            print("="*80 + "\n")
            break
        
        else:
            print("\n[AVISO] Opcion invalida. Por favor selecciona 1-6.")
        
        input("\nPresiona Enter para continuar...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSistema interrumpido. Hasta luego!")
    except Exception as e:
        print(f"\n[ERROR] Error inesperado: {e}")
