import json
import subprocess
import tempfile
from typing import Union
import pydantic
import inspect

import statextract.typedefs
import statextract.agent.stats_extractor_agent

def get_pydantic_models(module):
    models = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, pydantic.BaseModel):
            models.append(obj)
    return models

# import fastapi

# # create fake app
# app = fastapi.FastAPI()

# # construct 

# @app.get("/", response_model=tuple[*(get_pydantic_models(statextract.typedefs) + get_pydantic_models(statextract.agent.stats_extractor_agent))]) # type: ignore
# async def root():
#     return {"message": "Hello World"}




# def generate_openapi_schema(models: list[pydantic.BaseModel]):
#     """Generate an OpenAPI schema for the given Pydantic models. No services or anything, just 'components'.'schemas'."""
#     schemas = {}
#     for model in models:
#         schema = model.model_json_schema()
#         schemas[model.__name__] = schema
    
#     openapi_schema = {
#         "openapi": "3.0.0",
#         "info": {
#             "title": "Pydantic Models Schema",
#             "version": "1.0.0"
#         },
#         "components": {
#             "schemas": schemas
#         }
#     } 
    
#     return openapi_schema

if __name__ == "__main__":
    # models = get_pydantic_models(statextract.typedefs) + get_pydantic_models(statextract.agent.stats_extractor_agent)
    # openapi_schema = generate_openapi_schema(models)
    
    # print(json.dumps(openapi_schema, indent=2))
    
    from statextract.server import app
    
    openapi_schema = app.openapi()
    #dump to json
    print(json.dumps(openapi_schema, indent=2))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as f:
        json.dump(openapi_schema, f)
        f.flush()
    command = f"npx @hey-api/openapi-ts -i {f.name} -o frontend/src/generated_types -c @hey-api/client-fetch"
    subprocess.run(command, shell=True)