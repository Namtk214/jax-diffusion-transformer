import concurrent.futures
import pickle
import time
import os

def parent_dir(path: str) -> str:
    if "/" not in path:
        return "."
    return path.rsplit("/", 1)[0]

def name(path: str) -> str:
    return path.rsplit("/", 1)[1]


class Checkpoint:
    def __init__(self, filename, parallel: bool = False, default_name: str = "checkpoint.pkl"):
        self._filename = filename
        self._default_name = default_name
        self._values = {}
        self._parallel = parallel
        if self._parallel:
            self._worker = concurrent.futures.ThreadPoolExecutor(1, "checkpoint")
            self._promise = None

    def _resolve_filename(self, filename=None) -> str:
        """
        Trả về đường dẫn FILE thực sự sẽ được dùng để save/load.
        - Với gs://: giữ nguyên semantics như bản gốc (coi là file path).
        - Với path local:
            * Nếu có extension -> coi là file.
            * Nếu không extension -> coi là thư mục, gắn thêm default_name.
        """
        fname = filename or self._filename
        if "gs://" in fname:
            # Bản gốc coi đây là path FILE, không đụng đến.
            return fname

        # Nếu kết thúc bằng '/' thì coi như thư mục
        if fname.endswith(os.path.sep):
            return os.path.join(fname, self._default_name)

        root, ext = os.path.splitext(fname)
        if ext == "":
            # Không có extension -> coi là thư mục
            return os.path.join(fname, self._default_name)
        else:
            # Có extension -> coi là file
            return fname

    def __setattr__(self, name, value):
        # Giữ nguyên behaviour đặc biệt cho exists/save/load
        if name in ("exists", "save", "load"):
            return super().__setattr__(name, value)
        if name.startswith("_"):
            return super().__setattr__(name, value)
        self._values[name] = value

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._values[name]
        except KeyError:
            raise ValueError(name)

    def set_model(self, model):
        for key in model.__dict__.keys():
            data = getattr(model, key)
            if hasattr(data, "save") or key == "config":
                self._values[key] = getattr(model, key)

    def save(self, filename=None, keys=None):
        assert self._filename or filename
        filename = filename or self._filename
        resolved = self._resolve_filename(filename)
        print(f"Writing checkpoint: {resolved}")
        if self._parallel:
            self._promise and self._promise.result()
            self._promise = self._worker.submit(self._save, resolved, keys)
        else:
            self._save(resolved, keys)

    def _save(self, resolved_filename, keys):
        keys = tuple(self._values.keys() if keys is None else keys)
        assert all([not k.startswith("_") for k in keys]), keys
        data = {
            k: (self._values[k].save() if k != "config" else self._values[k])
            for k in keys
        }
        data["_timestamp"] = time.time()
        content = pickle.dumps(data)

        if "gs://" in resolved_filename:
            import tensorflow as tf

            tf.io.gfile.makedirs(parent_dir(resolved_filename))
            tmp = resolved_filename + ".tmp"
            with tf.io.gfile.GFile(tmp, "wb") as f:
                f.write(content)
            # Overwrite an toàn
            tf.io.gfile.rename(tmp, resolved_filename, overwrite=True)
        else:
            os.makedirs(parent_dir(resolved_filename), exist_ok=True)
            tmp = resolved_filename + ".tmp"
            with open(tmp, "wb") as f:
                f.write(content)
            # atomic overwrite
            os.replace(tmp, resolved_filename)

        print("Wrote checkpoint.")

    def exists(self, filename=None) -> bool:
        """
        Kiểm tra xem checkpoint file đã tồn tại chưa.
        """
        resolved = self._resolve_filename(filename)
        if "gs://" in resolved:
            import tensorflow as tf

            return tf.io.gfile.exists(resolved)
        else:
            return os.path.exists(resolved)

    def load_as_dict(self, filename=None):
        """
        Load dữ liệu checkpoint (dict) từ file.
        """
        assert self._filename or filename
        filename = filename or self._filename
        resolved = self._resolve_filename(filename)

        if "gs://" in resolved:
            import tensorflow as tf

            with tf.io.gfile.GFile(resolved, "rb") as f:
                data = pickle.loads(f.read())
        else:
            with open(resolved, "rb") as f:
                data = pickle.loads(f.read())

        age = time.time() - data["_timestamp"]
        print(f"Loaded checkpoint from {age:.0f} seconds ago.")
        return data

    def load_model(self, model, filename=None):
        cp_dict = self.load_as_dict(filename)
        replace_dict = {}
        for key in model.__dict__.keys():
            if key in cp_dict and key != "config":
                replace_dict[key] = getattr(model, key).load(cp_dict[key])
        return model.replace(**replace_dict)
