import boto3
import json
from botocore.exceptions import ClientError


def list_bedrock_models():
    """Lists all available models in Amazon Bedrock"""
    bedrock = boto3.client('bedrock', region_name='us-east-1')

    try:
        response = bedrock.list_foundation_models()
        models = response.get('modelSummaries', [])

        print("\n" + "=" * 80)
        print("AVAILABLE MODELS IN AMAZON BEDROCK")
        print("=" * 80 + "\n")

        # Filter only models that support conversation and on-demand invocation
        chat_models = []
        
        # Models that require inference profiles (exclude these)
        excluded_models = [
            'anthropic.claude-sonnet-4',
            'anthropic.claude-opus-4',
        ]
        
        for idx, model in enumerate(models, 1):
            model_id = model.get('modelId', 'N/A')
            model_name = model.get('modelName', 'N/A')
            provider = model.get('providerName', 'N/A')
            inference_types = model.get('inferenceTypesSupported', [])

            # Filter chat/conversation models
            if any(keyword in model_id.lower() for keyword in ['claude', 'llama', 'mistral', 'titan']):
                # Exclude models that require inference profiles
                if any(excluded in model_id.lower() for excluded in excluded_models):
                    continue
                
                # Only include models that support ON_DEMAND inference
                if 'ON_DEMAND' in inference_types or not inference_types:
                    chat_models.append({
                        'index': len(chat_models) + 1,
                        'id': model_id,
                        'name': model_name,
                        'provider': provider
                    })
                    print(f"{len(chat_models)}. {model_name}")
                    print(f"   ID: {model_id}")
                    print(f"   Provider: {provider}")
                    print()

        if chat_models:
            print("💡 Note: Only models with on-demand support are shown.\n")
        
        return chat_models

    except ClientError as e:
        print(f"Error listing models: {e}")
        return []


def chat_with_bedrock(model_id, user_message):
    """Sends a message to a Bedrock model and gets the response"""
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

    try:
        if 'claude' in model_id.lower():
            # Claude v3 and above uses the messages format
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
                # Claude v2 uses the prompt format
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
            print(f"Model not supported in this demo: {model_id}")
            return None

        # Invoke the model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body,
            contentType='application/json',
            accept='application/json'
        )

        # Parse the response
        response_body = json.loads(response['body'].read())

        # Extract text according to the provider
        if 'claude' in model_id.lower():
            if 'claude-3' in model_id.lower():
                return response_body['content'][0]['text']
            else:
                # Claude v2 uses 'completion'
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
            print(f"Validation error: {e}")
            print("💡 Suggestion: This model may require inference profiles or may not be available in your region.")
            print("   Try another model from the list.")
        elif error_code == 'AccessDeniedException':
            print(f"Access error: {e}")
            print("💡 Suggestion: Verify that your account has access to this model in AWS Bedrock.")
        else:
            print(f"Error invoking model: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def main():
    """Main demo function"""
    print("\n🤖 AMAZON BEDROCK CONVERSATION DEMO 🤖\n")

    # Step 1: List available models
    models = list_bedrock_models()

    if not models:
        print("No available models found.")
        return

    # Step 2: Select a model
    print("=" * 80)
    while True:
        try:
            selection = int(input(f"\nSelect a model (1-{len(models)}): "))
            if 1 <= selection <= len(models):
                selected_model = models[selection - 1]
                break
            else:
                print(f"Please select a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")

    print(f"\n✅ Selected model: {selected_model['name']}")
    print(f"   ID: {selected_model['id']}\n")

    # Step 3: Start conversation
    print("=" * 80)
    print("CONVERSATION (type 'exit' or 'quit' to end)")
    print("=" * 80 + "\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ['salir', 'exit', 'quit']:
            print("\n👋 Goodbye!")
            break

        if not user_input.strip():
            continue

        print("\n🤖 Assistant: ", end="", flush=True)
        response = chat_with_bedrock(selected_model['id'], user_input)

        if response:
            print(response)
        else:
            print("Could not get a response.")

        print()


if __name__ == "__main__":
    main()
