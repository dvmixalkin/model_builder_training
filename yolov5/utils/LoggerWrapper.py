from logging import StreamHandler
import time
from .Client import ws_client
import json
# class ProgressKeeper:
#     def __init__(self, epochs):
#         self.epoch = 0
#         self.epochs = epochs
#         self.trn_images = 
#     def

class SocketHandler(StreamHandler, ws_client):

    def __init__(self, 
                 server, 
                 port,
                 on_message_callback=None):

        StreamHandler.__init__(self)

        ws_client.__init__(self, 
                           channels_list=[], 
                           on_message_callback=None,
                           server=server,
                           port=port)

    def emit(self, record):
        
        msgtype = "error"
        if record.name == 'websocket':
            return
        if record.levelname == "INFO":
            try:
                msgdict = json.loads(record.msg)
                if "processed_images" in msgdict.keys():
                    msgtype = "progress"
                elif "metrics/precision" in msgdict.keys():
                    msgtype = "metrics"
            except: pass


        message = {"type": msgtype,
                   "data":json.dumps({"error":record.msg}) if msgtype=="error" else record.msg,
                   "date":time.time()}

        channel = record.name
        self.send(channel, message)



if __name__ == "__main__":
    '''DEMO'''
    import logging
    
    soha = SocketHandler(server='192.168.108.118',
                         port='8008')
    # Logger name = task name
    test_logger = logging.getLogger("test_logger")
    test_logger.setLevel(logging.INFO)
    test_logger.addHandler(soha)
    
    i = 0
    while True:

        test_logger.info(f"Ya rabotaju uje primerno {i} sekund", "04c3103a-7c11-41eb-a169-bd06f7e47209")
        time.sleep(1)
        i+=1