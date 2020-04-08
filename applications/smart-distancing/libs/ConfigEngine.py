#!/usr/bin/python3
import configparser
import threading

class ConfigEngine:
    def __init__(self, config_path = './config-skeleton.ini'):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        self.config_file_path = config_path
        self.lock = threading.Lock()
        # For dynamic and cross-chapter flexible parameters: 
        self.config._interpolation = configparser.ExtendedInterpolation()
        self.section_options_dict = {}
        self._load()
    
    def set_config_file(self, path):
        self.lock.acquire()
        try:
            self.config.clear()
            self.config_file_path = path
            self._load()
        finally:
            self.lock.release()

    def _load(self):
        self.config.read(self.config_file_path)
        for section in self.config.sections():
            self.section_options_dict[section] = {}
            options = self.config.options(section)
            for option in options:
                try:
                    val = self.config.get(section, option)
                    self.section_options_dict[section][option] = val
                    if val == -1:
                        print("skip: %s" % option)
                except:
                    print("exception on %s!" % option)
                    self.section_options_dict[section][option] = None

    def save(self, path):
        self.lock.acquire()
        try:
            file_obj = open(path, "w")
            self.config.write(file_obj)
            file_obj.close()
        finally:
            self.lock.release()

    def get_section_dict(self, section):
        return self.section_options_dict[section]
        
    def get_boolean(self, section, option):
        result = None
        self.lock.acquire()
        try:
            result = self.config.getboolean(section, option)
        finally:
            self.lock.release()

    def toggle_boolean(self, section, option):
        self.lock.acquire()
        try:
            val = self.config.getboolean(section, option)
            self.config.set(section, option, str(not val))
        finally:
            self.lock.release()
    
    def set_option_in_section(self, section, option, value):
        self.lock.acquire()
        try:
            self.config.set(section, option, value)
        finally:
            self.lock.release()
