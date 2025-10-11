import boto3
import json
from botocore.exceptions import ClientError


def list_bedrock_models():
    bedrock = boto3.client('bedrock', region_name='us-east-1')

    try:
        response = bedrock.list_foundation_models()
        models = response.get('modelSummaries', [])

        print("\n" + "=" * 80)
        print("MODELOS DISPONIBLES EN AMAZON BEDROCK")
        print("=" * 80 + "\n")

        # Modelos recomendados que funcionan bien (sin requerir perfiles de inferencia)
        recommended_models = [
            {
                'index': 1,
                'id': 'anthropic.claude-3-sonnet-20240229-v1:0',
                'name': 'Claude 3 Sonnet (Recomendado)',
                'provider': 'Anthropic'
            },
            {
                'index': 2,
                'id': 'anthropic.claude-3-haiku-20240307-v1:0',
                'name': 'Claude 3 Haiku (Rápido)',
                'provider': 'Anthropic'
            },
            {
                'index': 3,
                'id': 'anthropic.claude-v2:1',
                'name': 'Claude v2.1 (Estable)',
                'provider': 'Anthropic'
            },
            {
                'index': 4,
                'id': 'meta.llama3-1-70b-instruct-v1:0',
                'name': 'Llama 3.1 70B Instruct',
                'provider': 'Meta'
            },
            {
                'index': 5,
                'id': 'mistral.mixtral-8x7b-instruct-v0:1',
                'name': 'Mixtral 8x7B Instruct',
                'provider': 'Mistral AI'
            }
        ]

        for model in recommended_models:
            print(f"{model['index']}. {model['name']}")
            print(f"   ID: {model['id']}")
            print(f"   Proveedor: {model['provider']}")
            print()

        print("⚠️  Nota: Se muestran solo modelos probados y funcionales.")
        print("   Los modelos más nuevos pueden requerir configuración adicional.\n")

        return recommended_models

    except ClientError as e:
        print(f"Error al listar modelos: {e}")
        return []


def chat_with_bedrock(model_id, user_message):
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

    try:
        if 'claude' in model_id.lower():
            # Claude v3 y superior usa el formato de mensajes
            if 'claude-3' in model_id.lower():
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": user_message
                        }
                    ]
                })
            else:
                # Claude v2 usa el formato de prompt
                body = json.dumps({
                    "prompt": f"\n\nHuman: {user_message}\n\nAssistant:",
                    "max_tokens_to_sample": 1000,
                    "temperature": 0.7,
                    "top_p": 0.9
                })
        elif 'titan' in model_id.lower():
            body = json.dumps({
                "inputText": user_message,
                "textGenerationConfig": {
                    "maxTokenCount": 1000,
                    "temperature": 0.7,
                    "topP": 0.9
                }
            })
        elif 'llama' in model_id.lower():
            body = json.dumps({
                "prompt": f"<s>[INST] {user_message} [/INST]",
                "max_gen_len": 1000,
                "temperature": 0.7,
                "top_p": 0.9
            })
        elif 'mistral' in model_id.lower():
            body = json.dumps({
                "prompt": f"<s>[INST] {user_message} [/INST]",
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 0.9
            })
        else:
            print(f"Modelo no soportado en este demo: {model_id}")
            return None

        # Invocar el modelo
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body,
            contentType='application/json',
            accept='application/json'
        )

        # Parsear la respuesta
        response_body = json.loads(response['body'].read())

        # Extraer el texto según el proveedor
        if 'claude' in model_id.lower():
            if 'claude-3' in model_id.lower():
                return response_body['content'][0]['text']
            else:
                # Claude v2 usa 'completion'
                return response_body.get('completion', 'No response')
        elif 'titan' in model_id.lower():
            return response_body['results'][0]['outputText']
        elif 'llama' in model_id.lower():
            return response_body.get('generation', 'No response')
        elif 'mistral' in model_id.lower():
            return response_body.get('outputs', [{}])[0].get('text', 'No response')

        return str(response_body)

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'ValidationException':
            print(f"Error de validación: {e}")
            print("💡 Sugerencia: Este modelo puede requerir perfiles de inferencia o no estar disponible en tu región.")
            print("   Intenta con otro modelo de la lista.")
        elif error_code == 'AccessDeniedException':
            print(f"Error de acceso: {e}")
            print("💡 Sugerencia: Verifica que tu cuenta tenga acceso a este modelo en AWS Bedrock.")
        else:
            print(f"Error al invocar el modelo: {e}")
        return None
    except Exception as e:
        print(f"Error inesperado: {e}")
        return None


def main():
    """Función principal del demo"""
    print("\n🤖 DEMO DE CONVERSACIÓN CON AMAZON BEDROCK 🤖\n")

    # Paso 1: Listar modelos disponibles
    models = list_bedrock_models()

    if not models:
        print("No se encontraron modelos disponibles.")
        return

    # Paso 2: Seleccionar un modelo
    print("=" * 80)
    while True:
        try:
            selection = int(input(f"\nSelecciona un modelo (1-{len(models)}): "))
            if 1 <= selection <= len(models):
                selected_model = models[selection - 1]
                break
            else:
                print(f"Por favor, selecciona un número entre 1 y {len(models)}")
        except ValueError:
            print("Por favor, ingresa un número válido")

    print(f"\n✅ Modelo seleccionado: {selected_model['name']}")
    print(f"   ID: {selected_model['id']}\n")

    # Paso 3: Iniciar conversación
    print("=" * 80)
    print("CONVERSACIÓN (escribe 'salir' para terminar)")
    print("=" * 80 + "\n")

    while True:
        user_input = input("Tú: ")

        if user_input.lower() in ['salir', 'exit', 'quit']:
            print("\n👋 ¡Hasta luego!")
            break

        if not user_input.strip():
            continue

        print("\n🤖 Asistente: ", end="", flush=True)
        response = chat_with_bedrock(selected_model['id'], user_input)

        if response:
            print(response)
        else:
            print("No se pudo obtener una respuesta.")

        print()


if __name__ == "__main__":
    main()
