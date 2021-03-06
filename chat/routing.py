from django.conf.urls import url
from . import consumers

websocket_urlpatterns = [
    url(r"^ws/chat/(?P<room_name>[^/]+)/$", consumers.ChatConsumer),
    url(r"^ws/train/$", consumers.TrainConsumer),
    url(r"^ws/authentication/$", consumers.AuthenticationConsumer),
    url(r"^ws/calibrate/$", consumers.CalibrateConsumer),
    url(r"^ws/screen/$", consumers.ScreenConsumer),
]
