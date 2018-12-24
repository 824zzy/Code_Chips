import tornado.web
from handlers import frontend_handler
from handlers import probe_handler
from handlers import websocket_handler
import config

class Application(tornado.web.Application):
    """ Application inherited from tornado.
    """
    def __init__(self):
        handlers = [
            (r'/', websocket_handler.IndexPageHandler),
            (r'/probe_list', frontend_handler.Get_probe_list),
            (r'/submit_task', frontend_handler.Submit_task),
            (r'/show_tasks', frontend_handler.Show_tasks),
            (r'/display_task', frontend_handler.Display_task),
            (r'/send_task', probe_handler.Send_task),
            (r'/receive_mac', probe_handler.Receive_mac),
            (r'/receive_data', probe_handler.Receive_data),
            (r'/ws', websocket_handler.WebSocketHandler),
        ]
        super(Application, self).__init__(handlers, **config.settings)

if __name__ == "__main__":
    a = Application()


    
