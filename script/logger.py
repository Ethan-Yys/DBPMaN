# encoding=utf-8
import logging
import sys


# logger = logging.getLogger("tensorflow")
# 貌似tensorflow的logger默认就有一个StreamHandler了
# 所以，首先判断len(logger.handlers)是否为1
# 如果为1的话， 说明只有默认的StreamHandler,
# 那么先清空handlers,然后再加入指定格式(formatter)的StreamHandler和FileHandler

def set_logger():
    logger = logging.getLogger("tensorflow")
    if len(logger.handlers) == 1:
        logger.handlers = []
        logger.setLevel(logging.INFO)  # 去掉该行线上日志没有打印出来

        formatter = logging.Formatter(
            "%(asctime)s - [%(filename)s:%(lineno)d] - %(name)s - %(levelname)s - %(message)s")

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        fh = logging.FileHandler('tensorflow.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
