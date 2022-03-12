
class LOG():
    LOG_SILENT = 0
    LOG_ERR = 1
    LOG_INFO = 2
    LOG_DEBUG = 3
    def __init__(self, log_level):
        self.__l = log_level

    def err(self, *args):
        if self.__l >= self.LOG_ERR:
            print(*args)

    def info(self, *args):
        if self.__l >= self.LOG_INFO:
            print(*args)

    def dbg(self, *args):
        if self.__l >= self.LOG_DEBUG:
            print(*args)

    def set_level(self, level):
        self.__l = level
