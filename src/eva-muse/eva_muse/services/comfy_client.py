import json
import uuid
import urllib.request
import urllib.parse
import asyncio
import websockets
import logging
from typing import Dict, Any, Optional
from shared import get_settings

logger = logging.getLogger(__name__)

class ComfyUIClient:
    """
    Client Python pour interagir avec le serveur ComfyUI local.
    Sert de pont entre the Muse (LLM) et le moteur de rendu (Stable Diffusion / FLUX).
    """
    
    def __init__(self, server_address: Optional[str] = None):
        self.settings = get_settings()
        # Read ComfyUI address from settings (set via COMFYUI_HOST/COMFYUI_PORT env vars in docker-compose)
        default = f"{self.settings.comfyui_host}:{self.settings.comfyui_port}"
        self.server_address = server_address or default
        self.client_id = str(uuid.uuid4())

        
    def _queue_prompt(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Envoie le graphe (workflow) au serveur ComfyUI."""
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def _get_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        """Récupère l'image générée depuis le serveur."""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
            return response.read()

    def _get_history(self, prompt_id: str) -> Dict[str, Any]:
        with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())

    async def generate_from_workflow(self, workflow_json: Dict[str, Any]) -> list[bytes]:
        """
        Exécute un workflow ComfyUI complet et attend le résultat via WebSockets.
        Retourne une liste contenant les octets (bytes) des images/vidéos générées.
        """
        prompt_id = self._queue_prompt(workflow_json)['prompt_id']
        output_images = []
        
        uri = f"ws://{self.server_address}/ws?clientId={self.client_id}"
        logger.info(f"Connexion WebSocket à ComfyUI ({uri}) pour le job {prompt_id}...")
        
        async with websockets.connect(uri) as websocket:
            while True:
                out = await websocket.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message['type'] == 'executing':
                        data = message['data']
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            # Fin de l'exécution
                            break
                        
        history = self._get_history(prompt_id)[prompt_id]
        
        # Extraction des résultats (images sauvegardées par les noeuds "Save Image")
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                for image in node_output['images']:
                    image_data = self._get_image(image['filename'], image['subfolder'], image['type'])
                    output_images.append(image_data)
                    
        return output_images

    async def generate_video_from_workflow(self, workflow_json: Dict[str, Any]) -> bytes | None:
        """
        Executes a VHS (VideoHelperSuite) workflow in ComfyUI and returns the video bytes.
        Looks for 'videos' or 'gifs' in node outputs instead of 'images'.
        """
        prompt_id = self._queue_prompt(workflow_json)['prompt_id']

        uri = f"ws://{self.server_address}/ws?clientId={self.client_id}"
        logger.info(f"Waiting for video output from ComfyUI (job {prompt_id})...")

        async with websockets.connect(uri) as websocket:
            while True:
                out = await websocket.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message['type'] == 'executing':
                        data = message['data']
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            break

        history = self._get_history(prompt_id)[prompt_id]

        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            # VHS outputs videos under 'videos' or 'gifs' key
            for key in ('videos', 'gifs', 'images'):
                if key in node_output:
                    for item in node_output[key]:
                        return self._get_image(item['filename'], item.get('subfolder', ''), item.get('type', 'output'))

        return None

