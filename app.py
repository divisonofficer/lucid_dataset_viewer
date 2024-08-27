from flask import Flask, send_file, request, Response
import os
import json
import numpy as np
from typing import Optional

app = Flask(__name__)


class Frame:
    path: str
    scene: str
    path_prv: Optional[str]
    path_next: Optional[str]
    raw: dict[str, np.ndarray]
    post: dict[str, np.ndarray]

    def check_post_property(self):
        properties = [
            "depth",
            "disparity",
        ]
        d = {}
        for p in properties:
            d[p] = p in self.post if hasattr(self.post, "post") else False
        return d

    def check_file_property(self):
        properties = [
            "left_tonemapped.png",
            "right_tonemapped.png",
            "left_rectified.png",
            "right_rectified.png",
            "depth.png",
            "disparity.png",
            "lidar_plot.png",
        ]
        d = {}
        for p in properties:
            d[p] = os.path.exists(os.path.join(self.path, p))
        return d

    def property_status_dict(self):
        return {**self.check_post_property(), **self.check_file_property()}

    def __dict__(self):
        return {
            "path": self.path,
            "scene": self.scene,
            "path_prv": self.path_prv,
            "path_next": self.path_next,
            "property_status": self.property_status_dict(),
        }


class State:
    root_path: str
    scene_folders: list
    frame_map: dict[str, Frame] = {}

    def __init__(self):
        pass

    def load_from_file(self):
        file_path = os.path.join(self.root_path, "lucid_root.lucid")
        if os.path.exists(file_path):
            try:
                data = json.load(open(file_path))
                if "scene_folders" in data:
                    self.scene_folders = data.get("scene_folders")
            except Exception as e:
                pass

    def save_to_file(self):
        file_path = os.path.join(self.root_path, "lucid_root.lucid")
        json.dump(self.__dict__(), open(file_path, "w"))

    def __dict__(self):
        return {"root_path": self.root_path, "scene_folders": self.scene_folders}


state = State()


def read_frame(scene_folder: str, frame_folder: str):
    if frame_folder in state.frame_map:
        return state.frame_map[frame_folder]
    raw_npz = os.path.join(frame_folder, "raw.npz")
    frame = Frame()
    frame.path = frame_folder
    frame.scene = scene_folder
    frame.raw = {**np.load(raw_npz)}

    scene_frames = [
        x for x in state.scene_folders if x["scene_folder"] == scene_folder
    ][0]["frame_folders"]
    scene_frames.sort()
    frame_index = scene_frames.index(frame_folder)
    frame.path_prv = scene_frames[frame_index - 1] if frame_index > 0 else None
    frame.path_next = (
        scene_frames[frame_index + 1] if frame_index < len(scene_frames) - 1 else None
    )

    if os.path.exists(os.path.join(frame_folder, "post.npz")):
        post_npz = os.path.join(frame_folder, "post.npz")
        frame.post = {**np.load(post_npz)}

    state.frame_map[frame_folder] = frame
    return frame


def is_frame_folder(cur_dir: str):
    raw_npz = os.path.join(cur_dir, "raw.npz")
    post_npz = os.path.join(cur_dir, "post.npz")
    if os.path.exists(raw_npz) and os.path.exists(post_npz):
        return True
    return False


def is_scene_folder(cur_dir: str):
    sub_dirs = os.listdir(cur_dir)
    scene_folder_list = []
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(cur_dir, sub_dir)
        if is_frame_folder(sub_dir_path):
            scene_folder_list.append(sub_dir_path)
    if len(scene_folder_list) < 1:
        return None
    return {
        "scene_folder": cur_dir,
        "frame_folders": scene_folder_list,
        "frame_count": len(scene_folder_list),
    }


def get_scene_folders(cur_dir: str, depth=0):
    scene_foler_info = is_scene_folder(cur_dir)
    if scene_foler_info is not None:
        return [scene_foler_info]
    scene_folders = []
    if depth > 3:
        return []
    for sub_dir in os.listdir(cur_dir):
        sub_dir_path = os.path.join(cur_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            scene_folders += get_scene_folders(sub_dir_path, depth + 1)
    return scene_folders


@app.route("/", methods=["GET"])
def index():
    return send_file("template/index.html")


@app.route("/path/root/validate", methods=["POST"])
def validate_root_path():
    if hasattr(state, "root_path"):
        return Response(
            status=201,
            response=json.dumps({"path": state.root_path}),
            headers={"Content-Type": "application/json"},
        )
    path = request.json.get("path")

    path_f = os.path.join(path, "lucid_root.lucid")

    if os.path.exists(path_f):
        state.root_path = path
    elif os.path.exists(os.path.join("/bean/lucid", "lucid_root.lucid")):
        state.root_path = os.path.join("/bean/lucid")
    if hasattr(state, "root_path"):
        state.load_from_file()
        return Response(
            status=200 if state.root_path == path else 201,
            response=json.dumps({"path": state.root_path}),
            headers={"Content-Type": "application/json"},
        )

    return Response(status=404)


import base64


@app.route("/path/scene/<scene_folder_encoded>/main", methods=["GET"])
def get_scene_main(scene_folder_encoded):
    scene_folder = base64.b64decode(scene_folder_encoded).decode("utf-8")
    if not os.path.exists(scene_folder):
        return Response(status=404)
    return send_file(os.path.join("template", "scene_main.html"))


@app.route("/path/scene/<scene_folder_encoded>/info", methods=["GET"])
def get_scene_info(scene_folder_encoded):
    scene_folder = base64.b64decode(scene_folder_encoded).decode("utf-8")
    if not os.path.exists(scene_folder):
        return Response(status=404)

    scene_folder = [
        x for x in state.scene_folders if x["scene_folder"] == scene_folder
    ][0]

    return {"scene_folder": scene_folder}


@app.route("/path/scene/<scene_folder_encoded>/<frame_folder_encoded>", methods=["GET"])
def get_frame_view(scene_folder_encoded, frame_folder_encoded):
    scene_folder = base64.b64decode(scene_folder_encoded).decode("utf-8")
    frame_folder = base64.b64decode(frame_folder_encoded).decode("utf-8")
    if not os.path.exists(frame_folder):
        return Response(status=404)
    return send_file(os.path.join("template", "frame_main.html"))


@app.route(
    "/path/scene/<scene_folder_encoded>/<frame_folder_encoded>/png/<file>",
    methods=["GET"],
)
def get_frame_property_png(scene_folder_encoded, frame_folder_encoded, file):
    frame_folder = base64.b64decode(frame_folder_encoded).decode("utf-8")
    return send_file(os.path.join(frame_folder, file))


@app.route(
    "/path/scene/<scene_folder_encoded>/<frame_folder_encoded>/info", methods=["GET"]
)
def get_frame_info(scene_folder_encoded, frame_folder_encoded):
    scene_folder = base64.b64decode(scene_folder_encoded).decode("utf-8")
    frame_folder = base64.b64decode(frame_folder_encoded).decode("utf-8")
    if not os.path.exists(frame_folder):
        return Response(status=404)

    frame = read_frame(scene_folder, frame_folder)
    return Response(
        status=200,
        response=json.dumps(frame.__dict__()),
        headers={"Content-Type": "application/json"},
    )


@app.route("/path/scene/list", methods=["GET"])
def get_scene_list():
    if hasattr(state, "scene_folders"):
        return {"scene_folders": state.scene_folders}
    scene_folders = get_scene_folders(state.root_path)
    state.scene_folders = scene_folders
    state.save_to_file()
    return {"scene_folders": scene_folders}


if __name__ == "__main__":
    app.run(debug=True)
