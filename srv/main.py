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
    res = ezkl.gen_settings(loaded_onnxmodel, settings_path)
    assert res == True
    res = await ezkl.calibrate_settings(loaded_inputdata, loaded_onnxmodel, settings_path, "resources")
    assert res == True

    # srs path
    settingsDict = load_json_file(settings_path)
    # print(settingsDict)
    print(settingsDict["run_args"]["logrows"])


    res = await ezkl.get_srs(settings_path, settingsDict["run_args"]["logrows"],
                    srs_path=srs_path)
    assert res == True

    res = ezkl.compile_circuit(loaded_onnxmodel, compiled_model_path, settings_path)
    assert res == True
    # srs path
    res = ezkl.get_srs(settings_path, settingsDict["run_args"]["logrows"],
                    srs_path=srs_path)
    assert res == True

    res = ezkl.gen_witness(loaded_inputdata, compiled_model_path, witness_path)
    assert os.path.isfile(witness_path)
    
    res = ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
        srs_path,
    )

    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)

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

    #print(proof)
    assert os.path.isfile(proof_path)

    # verify our proof
    print("verifying proof...")
    res = ezkl.verify(
            proof_path,
            settings_path,
            vk_path,
            srs_path,
        )

    assert res == True
    print("verified")


    sol_code_path = os.path.join('Verifier.sol')
    abi_path = os.path.join('Verifier.abi')

    res = ezkl.create_evm_verifier(
            vk_path,
            srs_path,
            settings_path,
            sol_code_path,
            abi_path
        )

    assert res == True
    assert os.path.isfile(sol_code_path)

    # try:
    #     running = True
    #     print("Generating EVM Verifier")
    #     res = ezkl.gen_settings(loaded_onnxmodel, settings_path)
    #     assert res == True
        
    #     res = await ezkl.calibrate_settings(loaded_inputdata, model_path, settings_path, "resources")  # Optimize for resources

    #     # p = subprocess.run([
    #     #         ezkl,
    #     #         "--bits=16",
    #     #         "-K=17",
    #     #         "create-evm-verifier",
    #     #         "-D", os.path.join(os.getcwd(), loaded_inputdata),
    #     #         "-M", os.path.join(os.getcwd(), loaded_onnxmodel),
    #     #         "--deployment-code-path", os.path.join(os.getcwd(), "generated", loaded_proofname + ".code"),
    #     #         "--params-path=" + os.path.join(os.getcwd(), "kzg.params"),
    #     #         "--vk-path", os.path.join(os.getcwd(), "generated", loaded_proofname + ".vk"),
    #     #         "--sol-code-path", os.path.join(os.getcwd(), "generated", loaded_proofname + ".sol"),
    #     #     ],
    #     #     capture_output=True,
    #     #     text=True
    #     # )
    #     print("Done generating EVM Verifier")
    #     running = False

        # return {
        #     # "stdout": p.stdout,
        #     # "stderr": p.stderr
        # }

    # except:
    #     running = False
    #     err = traceback.format_exc()
    #     return "Something bad happened! Please inform the server admin\n" + err, 500
    return { "message": "win"}


@app.get("/proofgen")
async def proofgen():
    proof_path = os.path.join('proving/test.pf')
    witness_path=os.path.join('proving/witness.json')
    compiled_model_path=os.path.join('proving/network.compiled')
    pk_path=os.path.join('proving/test.pk')
    srs_path=os.path.join('proving/kzg.srs')

    res = ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        proof_path,
        "single",
        srs_path=srs_path
    )
    #assert res == True
    assert os.path.isfile(proof_path)
    return True

@app.get("/verify")
async def verify():
    proof_path = os.path.join('proving/test.pf')
    settings_path = os.path.join('proving/settings.json')
    vk_path = os.path.join('proving/test.vk')
    srs_path = os.path.join('proving/kzg.srs')
    # VERIFY IT
    res = ezkl.verify(
            proof_path,
            settings_path,
            vk_path,
            srs_path=srs_path
        )

    assert res == True
    print("verified")
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