import os
import sys
import logging
import subprocess
import ezkl
import traceback
from dotenv import load_dotenv, find_dotenv
import uuid
import json

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi import FastAPI
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

import uvicorn
from pydantic import BaseModel

import numpy as np

class ModelInput(BaseModel):
    inputdata: str
    onnxmodel: str

load_dotenv(find_dotenv())

WEBHOOK_PORT = int(os.environ.get("PORT", 8080))  # 443, 80, 88 or 8443 (port need to be 'open')
WEBHOOK_LISTEN = '0.0.0.0' 

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = FastAPI()
#ezkl = "./ezkl/target/release/ezkl"

# allow cors
app.add_middleware(
    CORSMiddleware,
    #allow_origins=origins,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

# Class UploadOnnxModel(BaseModel):

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return None


"""
Generates evm verifier
"""
@app.get('/run/gen_evm_verifier')
async def gen_evm_verifier():
    # global loaded_inputdata
    # global loaded_onnxmodel
    # global loaded_proofname
    # global running

    loaded_inputdata="inputdata/soccertorchinput.json"
    loaded_onnxmodel="onnxmodel/soccertorch.onnx"
    loaded_proofname="inputdata_soccertorchinput+onnxmodel_soccertorch"
    running=False

    print(loaded_inputdata)
    print(loaded_onnxmodel)
    print(loaded_proofname)
    print(running)
    settings_path = os.path.join('settings.json')
    print(settings_path)
    srs_path = os.path.join('kzg.srs')
    print(srs_path)
    compiled_model_path = os.path.join('soccercircuit.compiled')
    pk_path = os.path.join('test.pk')
    vk_path = os.path.join('test.vk')
    witness_path = os.path.join('witness.json')


    # if loaded_inputdata is None or loaded_onnxmodel is None or loaded_proofname is None:
    #     return "Input Data or Onnx Model not loaded", 400
    # if running:
    #     return "Already running please wait for completion", 400
    # if os.path.exists(os.path.join(os.getcwd(), "generated", loaded_proofname + ".sol")) and \
    #     os.path.exists(os.path.join(os.getcwd(), "generated", loaded_proofname + ".code")):
    #     return "Verifier already exists", 400

    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "public"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "private" # private by default

    print("Generating settings...")
    res = ezkl.gen_settings(loaded_onnxmodel, settings_path)
    assert os.path.isfile(settings_path)
    assert res == True

    print("Calibrating settings...")
    res = await ezkl.calibrate_settings(loaded_inputdata, loaded_onnxmodel, settings_path, "resources")
    assert res == True
    
    print("Compiling circuit...")
    res = ezkl.compile_circuit(loaded_onnxmodel, compiled_model_path, settings_path)
    assert res == True

    # srs path
    settingsDict = load_json_file(settings_path)
    # print(settingsDict)
    print(settingsDict["run_args"]["logrows"])

    print("Generating SRS...")
    res = await ezkl.get_srs(settings_path, 
                            settingsDict["run_args"]["logrows"],
                            srs_path=srs_path
                            )
    assert res == True
    #res = ezkl.get_srs(srs_path, settingsDict["run_args"]["logrows"])# ## DEPRECATED
    
    # now generate the witness file 
    print("Generating witness...")
    res = ezkl.gen_witness(loaded_inputdata, compiled_model_path, witness_path)
    #assert res == True
    assert os.path.isfile(witness_path)
    

    print("Setting up key files...")
    res = ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
        srs_path=srs_path,
        witness_path=witness_path,
    )
    assert res == True

    # verifying the setup
    print("verifying setup complete...")
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)

    # Generate the proof
    # print("generating proof...")
    # proof_path = os.path.join('test.pf')

    # proof = ezkl.prove(
    #         witness_path,
    #         compiled_model_path,
    #         pk_path,
    #         proof_path,
    #         srs_path,
    #         "evm",
    #         "single",
    #         settings_path
    #     )

    # print(proof)
    # assert os.path.isfile(proof_path)

    # # verify our proof
    # print("verifying proof...")
    # res = ezkl.verify(
    #         proof_path,
    #         settings_path,
    #         vk_path,
    #         srs_path,
    #     )

    # assert res == True
    # print("verified")


    # sol_code_path = os.path.join('Verifier.sol')
    # abi_path = os.path.join('Verifier.abi')

    # res = ezkl.create_evm_verifier(
    #         vk_path,
    #         srs_path,
    #         settings_path,
    #         sol_code_path,
    #         abi_path
    #     )

    # assert res == True
    # assert os.path.isfile(sol_code_path)

    return { "message": "win"}


@app.post('/predict')
async def predict():
    # jsonpayload = await request.json()
    
    # results_dict = {}
    # for i, b in enumerate(jsonpayload):
    #     x = np.array([b]).astype("float32")
    #     onnx_pred = sess.run([output_name], {input_name: x})
        #score = onnx_pred[0][0][0]
        #results_dict[i] = score
    x= np.array([[
        63.0, 76.0, 56.0, 70.0, 27.0, 84.0, 
        77.0, 78.0, 75.0, 75.0, 45.0, 56.0, 
        61.0, 61.0, 68.0, 72.0, 72.0, 71.0, 
        76.0, 47.0, 65.0, 68.0, 74.0, 74.0, 
        67.0, 31.0, 55.0, 57.0, 75.0, 75.0, 
        76.0, 86.0, 93.0, 88.0, 64.0, 78.0, 
        61.0, 70.0, 77.0, 78.0, 82.0, 82.0, 
        65.0, 80.0, 85.0, 86.0, 73.0, 72.0, 
        61.0, 38.0, 65.0, 68.0, 88.0, 88.0, 
        85.0, 71.0, 83.0, 84.0, 80.0, 72.0
        ]]).astype("float32")
    
    alt = np.array([[
        90.0,90.0,90.0,90.0,90.0,90.0,
        90.0,90.0,90.0,90.0,90.0,90.0,
        90.0,90.0,90.0,90.0,90.0,90.0,
        90.0,90.0,90.0,90.0,90.0,90.0,
        90.0,90.0,90.0,90.0,90.0,90.0,

        80.0,80.0,80.0,80.0,80.0,80.0, 
        80.0,80.0,80.0,80.0,80.0,80.0, 
        80.0,80.0,80.0,80.0,80.0,80.0, 
        80.0,80.0,80.0,80.0,80.0,80.0, 
        80.0,80.0,80.0,80.0,80.0,80.0
        ]]).astype("float32")
    
    onnx_pred = sess.run([output_name], {input_name: x})
    print(onnx_pred)

    result = np.argmax(onnx_pred, axis=-1)
    print(result)

    if(result == 0):
        result = "1-0"
    elif(result == 2):
        result = "0-1"
    else:
        result = "1-0"

    # generate proof
    loaded_inputdata="inputdata/soccertorchinput.json"
    loaded_onnxmodel="onnxmodel/soccertorch.onnx"
    loaded_proofname="inputdata_soccertorchinput+onnxmodel_soccertorch"
    running=False

    print(loaded_inputdata)
    print(loaded_onnxmodel)
    print(loaded_proofname)
    print(running)
    settings_path = os.path.join('settings.json')
    print(settings_path)
    srs_path = os.path.join('kzg.srs')
    print(srs_path)
    compiled_model_path = os.path.join('soccercircuit.compiled')
    pk_path = os.path.join('test.pk')
    vk_path = os.path.join('test.vk')
    witness_path = os.path.join('witness.json')

    res = ezkl.gen_witness(loaded_inputdata, compiled_model_path, witness_path)
    assert os.path.isfile(witness_path)

    # Generate the proof
    print("generating proof...")
    proof_path = os.path.join('proof.json')

    proof = ezkl.prove(
            witness_path,
            compiled_model_path,
            pk_path,
            proof_path,
            srs_path,
            "single",
        )
    
    assert os.path.isfile(proof_path)

    # verify our proof via smart contracts


    #sorted_results = sorted(results_dict.items(), key=lambda x:x[1], reverse=True)
    #print(sorted_results)
    #bestconfigkey = sorted_results[0][0]
    #print(jsonpayload[bestconfigkey])

    #return JSONResponse(content=jsonable_encoder(jsonpayload[bestconfigkey]))
    return { "message": result}

@app.get("/verify")
async def verify():
    return True

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=WEBHOOK_LISTEN,
        port=WEBHOOK_PORT
    )