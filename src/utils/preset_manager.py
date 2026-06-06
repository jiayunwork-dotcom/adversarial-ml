import os
from typing import Dict, Any

from config import PRESET_DIR, PRESET_ATTACK_CONFIGS
from .helpers import save_json, load_json, generate_id


class PresetManager:
    def __init__(self):
        self.preset_file = os.path.join(PRESET_DIR, "attack_presets.json")
        self._init_default_presets()

    def _init_default_presets(self):
        if not os.path.exists(self.preset_file):
            presets = {}
            for key, config in PRESET_ATTACK_CONFIGS.items():
                presets[key] = {
                    "id": key,
                    "name": config["name"],
                    "attack": config["attack"],
                    "params": config["params"],
                    "created_at": "default"
                }
            save_json(presets, self.preset_file)

    def list_presets(self) -> Dict[str, Any]:
        return load_json(self.preset_file)

    def get_preset(self, preset_id: str) -> Dict[str, Any]:
        presets = self.list_presets()
        return presets.get(preset_id, {})

    def save_preset(self, name: str, attack: str, params: Dict[str, Any]) -> str:
        presets = self.list_presets()
        preset_id = generate_id()
        presets[preset_id] = {
            "id": preset_id,
            "name": name,
            "attack": attack,
            "params": params,
            "created_at": "custom"
        }
        save_json(presets, self.preset_file)
        return preset_id

    def delete_preset(self, preset_id: str) -> bool:
        presets = self.list_presets()
        if preset_id in presets and presets[preset_id]["created_at"] != "default":
            del presets[preset_id]
            save_json(presets, self.preset_file)
            return True
        return False
