import torch
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(os.pardir)

from django.apps import AppConfig
from few_shot_gaze.src.models import DTED

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#################################
# Load gaze network
#################################
ted_parameters_path = r'few_shot_gaze/demo/demo_weights/weights_ted.pth.tar'
maml_parameters_path = r'few_shot_gaze/demo/demo_weights/weights_maml'
k = 9
lr = 1e-5
steps = 5000
cnt = 0

from pathlib import Path

from authentication.facenet import loadFacenet
from authentication.liveness import *


class UsersConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'users'


class EyeConfig(AppConfig):
    name = 'eye'
    gaze_network = DTED(
        growth_rate=32,
        z_dim_app=64,
        z_dim_gaze=2,
        z_dim_head=16,
        decoder_input_c=32,
        normalize_3d_codes=True,
        normalize_3d_codes_axis=1,
        backprop_gaze_to_encoder=False,
    ).to(device)

    vanila_gaze_network = gaze_network

    #################################

    # Load DT-ED weights if available
    assert os.path.isfile(ted_parameters_path)
    print('> Loading: %s' % ted_parameters_path)
    ted_weights = torch.load(ted_parameters_path)
    if torch.cuda.device_count() == 1:
        if next(iter(ted_weights.keys())).startswith('module.'):
            ted_weights = dict([(k[7:], v) for k, v in ted_weights.items()])

    #####################################

    # Load MAML MLP weights if available
    full_maml_parameters_path = maml_parameters_path + '/%02d.pth.tar' % k
    assert os.path.isfile(full_maml_parameters_path)
    print('> Loading: %s' % full_maml_parameters_path)
    maml_weights = torch.load(full_maml_parameters_path)
    ted_weights.update({  # rename to fit
        'gaze1.weight': maml_weights['layer01.weights'],
        'gaze1.bias': maml_weights['layer01.bias'],
        'gaze2.weight': maml_weights['layer02.weights'],
        'gaze2.bias': maml_weights['layer02.bias'],
    })
    gaze_network.load_state_dict(ted_weights)


class LivenessConfig(AppConfig):
    name ='liveness'
    MODEL_PATH = Path('./authentication/models/liveness.model')
    model = loadLiveness(MODEL_PATH)


class FacenetConfig(AppConfig):
    name = 'facenet'
    WEIGHTS_PATH = Path('./authentication/models/facenet_weights.h5')
    model = loadFacenet(WEIGHTS_PATH)

