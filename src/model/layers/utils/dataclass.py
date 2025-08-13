from typing import Any
from dataclasses import is_dataclass, fields
from omegaconf import DictConfig, OmegaConf

def shallow_asdict(obj:Any)->dict:
    if is_dataclass(obj):
        return {f.name: getattr(obj, f.name) for f in fields(obj)}
    elif isinstance(obj, DictConfig):  
        return {k: v if not isinstance(v, DictConfig) else v for k, v in obj.items()}
    else:
        raise TypeError(f"Unsupported type for shallow_asdict: {type(obj)}")
    return obj

def safe_replace(obj: Any, **kwargs) -> Any:
    if is_dataclass(obj):
        field_names = {f.name for f in fields(obj)}
        new = copy.deepcopy(obj)
        for key, value in kwargs.items():
            if key in field_names:
                setattr(new, key, value)
        return new                
    elif isinstance(obj, DictConfig):
        return OmegaConf.merge(obj, OmegaConf.create(kwargs))
    else:
        raise TypeError(f"Unsupported type for safe_replace: {type(obj)}")
