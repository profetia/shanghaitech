import os
from dynaconf import Dynaconf

src_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(src_folder)

cfg: Dynaconf = Dynaconf(
    envvar_prefix="HORRAS",
    environments=["development", "production", "testing"],
    env_switcher="HORRAS_ENV",
    load_dotenv=True,
    dotenv_path=project_folder
)