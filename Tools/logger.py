# log the parameters and the results
import datetime

class Logger(object):
    # path: the path of the log file
    def __init__(self, path):
        self.path = path
    # log the parameters
    def log_param(self, config):
        with open(self.path,'a+') as logf:
            timeStr = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
            logf.writelines("[%s]-[Config]\n"%(timeStr))
            for i,item in enumerate(config.__dict__.items()):
                text = "[%d]-[parameters] %s--%s"%(i,item[0],str(item[1]))
                print(text)
                logf.writelines(text+'\n')
    # log the text
    def log_text(self, text):
        with open(self.path,'a+') as logf:
            logf.writelines(text)
    # log the results
    def record(self, **args):
        with open(self.path,'a+') as logf:
            timeStr = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
            text = f"[{timeStr}]"
            for key in args.keys():
                text += "-[{}]-[{}]".format(key, args[key])
            logf.writelines(text + "\n")
        return text