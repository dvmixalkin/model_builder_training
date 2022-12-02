# import sys
import websocket
import _thread
import time
import json


# websocket.enableTrace(False)
# import random


class ws_client:
    def __init__(self, channels_list=[], on_message_callback=None, server='192.168.108.118', port='8008'):
        # if len(channels_list)==0:
        #     print('Cant start without list of channels')
        #     raise
        self.channels = channels_list
        self.reconnect_time = 6
        self.connected = False
        self.message_list = []
        self.server_address = server
        self.server_port = port
        self.on_message_callback = on_message_callback if not (on_message_callback is None) else self.ws_message
        self.my_current_thread = _thread.start_new_thread(self.socket_handler, ())

    def socket_handler(self):

        self.ws = websocket.WebSocketApp(f"ws://{self.server_address}:{self.server_port}", on_open=self.ws_open,
                                         on_message=self.on_message_callback, on_close=self.ws_close)
        self.ws.run_forever()

    def ws_message(self, ws, message):
        print(f"Receive {message.decode('UTF-8')}")

    def run(self):
        while True:
            if self.connected:
                if len(self.message_list):
                    reversed_message_list = self.message_list
                    message = self.message_list.pop()
                    # print(message)
                    self.ws.send(message)

    def ws_open(self, ws):
        self.connected = True
        if (len(self.channels) == 1 and self.channels[0] != "Source"):
            self.ws.send(json.dumps({"connect": self.channels}).encode('utf-8'))
        _thread.start_new_thread(self.run, ())

    def ws_close(self, ws, status, message):
        print("Connection closed, try again")
        self.connected = False
        time.sleep(self.reconnect_time)
        self.socket_handler()
        # self.reconnect()

    def connect(self, channel):
        if not (channel in self.channels):
            self.channels.append(channel)
            self.message_list.append(json.dumps({"connect": self.channels}).encode('utf-8'))

    def disconnect(self, ws, channel):
        if channel in self.channels:
            self.channels.remove(channel)
            self.message_list.append(json.dumps({"disconnect": [channel]}).encode('utf-8'))

    def send(self, channel=None, message=None):
        if channel is None:
            print('Cant send message to empty channel')
            return
        if message is None:
            print(f'Cant send empty message to channel {channel}')
            return
        self.message_list.insert(0, json.dumps({'channel': channel,
                                                "message": message}))
        return True


if __name__ == '__main__':
    my_ws_client = ws_client()
    my_ws_client.connect("eb0bf715-5619-4733-91c4-86706e46ccad")
    while True:
        time.sleep(5)
