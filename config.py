import configparser

# Default path for system.ini
SYSTEM_INI_PATH = 'system.ini'

class Settings:
    def __init__(self):
        self.system = {}

    def load_settings(self):
        # Load settings from system.ini
        config = configparser.ConfigParser()
        config.read(SYSTEM_INI_PATH)

        # Populate settings.system from the INI file
        for section in config.sections():
            self.system[section] = dict(config.items(section))

        # Ensure environment secrets or YAML are not mixed
        # Environment variables and YAML loading logic go here...  

