import configparser


cfg = configparser.ConfigParser()

cfg["mypy-numba"] = {
    "ignore_missing_imports": "True"
}

with open("mypy.ini", "w") as file:
    cfg.write(file)
