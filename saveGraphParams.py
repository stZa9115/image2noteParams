import json
import os
import numpy as np


def to_list(x):
    """安全轉成 JSON 可用型別"""
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def save_graphic_json(exprs, image_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    (intensity, note, density,
     hue, saturation, value,
     time_axis) = exprs

    data = {
        "source_image": os.path.basename(image_path),
        "time_axis": to_list(time_axis),

        "geometry": {
            "intensity": to_list(intensity),
            "note_trend": to_list(note),
            "density": to_list(density)
        },

        "color": {
            "hue": to_list(hue),
            "saturation": to_list(saturation),
            "value": to_list(value)
        }
    }

    out_path = os.path.join(out_dir, 
        os.path.splitext(os.path.basename(image_path))[0] + ".json"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"graphic json written: {out_path}")
