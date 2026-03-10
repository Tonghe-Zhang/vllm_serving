
## How to use this repo to host a open-source VLM and listen to it from another machine:

### Install server and client dependencies:
* Install openai package in the client directory by running 
```bash
uv pip install ./vllm_serving/install_client_env.sh
```
* Install the server packages in the large remote server by running
```bash
uv pip install ./vllm_serving/install_server_env.sh
```
* Download the [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) on the remote server. 
|_vllm_serving
|_PretrainedModels
    |_Qwen3-VL-8B-Instruct
You can also download the thinking model

### Start vllm inference on a server machine:
On remote server, run: 
```bash
./vllm_serving/server/run_qwen3_vl_server.sh
```
or 
```bash
./vllm_serving/server/run_qwen3_vl_server_lean.sh
```
if your resource is constrained. 

### Talk to server from the client machine:

On client machine, 
listen to the running server through SSH channel:
```bash
ssh -N -L 8000:localhost:8000 SERVER_SSH_CONFIG_NAME
```
For example, 
```bash
ssh -N -L 8000:localhost:8000 LeCAR_4xRTX6000BlackWell_97GB
```
When you see that there is a port named 8000 on the client machine, you can now listen to the server. 

Then open another temrinal, and run, 
```bash
./vllm_serving/client/run_qwen3_vl_client.sh
```
or 
```bash
./vllm_serving/client/run_qwen3_vl_client_sys_prompt.sh
```
And then we will send local client's text and images to the server for vllm inference, and then the results send back from that server will be shown in the client's terminal. 