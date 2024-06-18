# Packet_KAI8
Packet Buddy using Kubernetes with 8 different SLM/LLM for multi-AI analysis 

## Getting Started

This system uses Kubernetes.

On Windows you can use WSL2 with Ubuntu and Docker Desktop. Then, inside Docker Desktop, enable Kubernetes. From there you simply need to apply the YAML configuration files to start Packet KAI8 and visit localhost:8501 to start chatting with your Packets. 

First clone the repository locally 

## Git stuff

```console
$ git clone https://github.com/automateyournetwork/Packet_KAI8.git
```

### WSL2 
Launch Powershell as admin and type: 

PS C:\Windows\system32>wsl --install

This will install WSL2 and Ubuntu; reboot. 

Ubuntu will launch on reboot - set your username and password

### Docker Desktop 
Download Docker Desktop - https://www.docker.com/products/docker-desktop/

#### Enabled Kubernets
In Docker Desktop - Settings - Kubernetes - toggle the On button

Wait for Kubernetes to start

## Starting Packet KAI8
In order run the following commands from Ubuntu: 

```console
$ kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/deployments/static/nvidia-device-plugin.yml
$ kubectl apply -f ollama-service.yaml
$ kubectl apply -f nginx-config.yaml
$ kubectl apply -f ollama-pod.yaml
```

## Stopping Packet KAI8 
```console
$ kubectl delete -f ollama-service.yaml
$ kubectl delete -f nginx-config.yaml
$ kubectl delete -f ollama-pod.yaml
```

## Adding more models 
As more models become available I will try to keep up and add them but if you wanted to add more models update the following files as follows: 

### nginx-config
Add a new upstream server - increment the port 

```yaml
      upstream mixtral_server {
        server 127.0.0.1:11441;
      }
```

Add a new server 

```yaml
        location /api/mixtral/generate {
          rewrite ^/api/mixtral/generate(.*)$ $1 break;
          proxy_pass http://mixtral_server;
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header X-Forwarded-Proto $scheme;
          proxy_read_timeout 600s;
          proxy_connect_timeout 600s;
          proxy_send_timeout 600s;
        }
```

### ollama-pod
Copy and add a new container service renaming the model as appropriate and incrementing the port matching the port from NGINX 

```yaml
  - name: ollama-mixtral
    image: ollama/ollama:latest
    command: ['sh', '-c', 'ollama start & sleep 20; ollama pull mixtral && tail -f /dev/null']
    ports:
    - containerPort: 11441
    env:
    - name: OLLAMA_MODEL_PATH
      value: /models/
    - name: OLLAMA_HOST
      value: "127.0.0.1:11441"
    - name: OLLAMA_KEEP_ALIVE
      value: "0"      
    resources:
      requests:
        memory: "2Gi"
        cpu: "1"
      limits:
        memory: "4Gi"
        cpu: "2"
    volumeMounts:
    - mountPath: /models
      name: model-storage
```

## VS Code 
I strongly recommend using VS Code with the Docker and Kubernetes extensions allowing you to explore and visualize and run the commands vis Ubuntu terminal. 