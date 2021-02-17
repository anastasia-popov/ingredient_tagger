import json
import boto3
import re

    
def parse_ingredient_string(ingr_string, tags):
    """
    Converts ingredient string to a dictionary of quantity, unit and 
    ingredient name according to the list of corresponding tags
    """
    words = ingr_string.split(' ')
    quantity = []
    ingredient = []
    unit = []
    tags = tags[:len(words)]
    for w, t in zip(words, tags):
        if t in [1, 2]:
            quantity.append(w.replace('$', ' '))
        elif t in [3, 4]:
            unit.append(w)
        elif t in [5, 6]:
            ingredient.append(w)
    return {'quantity': ' '.join(quantity), 'unit': ' '.join(unit), 'ingredient': ' '.join(ingredient)}
            

def lambda_handler(event, context):
    
    ingr_strings = json.loads(event['body'])
    
    #concatenate digits 
    ingr_strings = [re.sub(r'(\d+)\s+(\d)/(\d)', r'\1$\2/\3', s) for s in ingr_strings]
    
    runtime = boto3.Session().client('sagemaker-runtime')

    # Now we use the SageMaker runtime to invoke our endpoint
    response = runtime.invoke_endpoint(EndpointName = # The name of the endpoint to call
                                       ContentType = 'text/json',                 
                                       Body = json.dumps(ingr_strings).encode('utf-8'))

    # The response is an HTTP response whose body contains the result of our inference
    tag_lists = json.loads(response['Body'].read().decode('utf-8'))

    result = []
    
    for ingr_str, tag_list in zip(ingr_strings, tag_lists):
        result.append(parse_ingredient_string(ingr_str, tag_list))

    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body' : json.dumps(result)
    }

