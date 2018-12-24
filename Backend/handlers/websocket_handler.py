# -*- coding: utf-8 -*-
import tornado
import tornado.web
import tornado.websocket
import tornado.ioloop
from models.agent_model import Agent
import json
import os


all_agent = dict()

class WebSocketHandler(tornado.websocket.WebSocketHandler):

    def open(self, *args, **kwargs):
        print("websocket opened")
        # print(self.request.body)
        # print(type(self.request.body))
        # m_dict = json.loads(self.request.body.decode('utf8'))
        # print(m_dict)
        # mac = m_dict["mac"]
        # mac = self.get_argument("mac")

    
        

    def on_close(self):
        """call this function when server is closing
        """
        # del all_agent[mac]
        # self.write_message(str(mac) + " is closed.")

    def on_message(self, message):
        """call this function when receive message
        """
        # self.write_message("You said: " + message)
        print(message)
        print(type(message))
        msg = json.loads(message)
        if len(msg) == 1:
            all_agent[msg["mac"]] = self
            Agent.create_agent(msg["mac"])
        else:
            # try:
            print(msg)
            Agent.update_data_with_mac(msg['mac'], msg['rtt_avg'], msg['rtt_stddev'], msg['packet_loss'])
            # except e:
            #     print("something wrong")
            #     pass
            
        # print(msg['mac'], msg['rtt_avg'], msg['rtt_stddev'], msg['packet_loss'])
        # try:
        #     Agent.update_data_with_mac(msg['mac'], msg['rtt_avg'], msg['rtt_stddev'], msg['packet_loss'])
        # except e:
        #     print("something wrong")
        #     pass

    # def callback(self, count):
    #     self.write_message('{"inventorycount":"%s"}' % count)


class IndexPageHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("websockets.html")


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r'/', IndexPageHandler),
            (r'/ws', WebSocketHandler)
        ]

        settings = {
            'template_path': 'static'
        }
        tornado.web.Application.__init__(self, handlers, **settings)


if __name__ == '__main__':
    ws_app = Application()
    server = tornado.httpserver.HTTPServer(ws_app)
    server.listen(50112)
    tornado.ioloop.IOLoop.instance().start()

