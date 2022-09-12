# Copyright (c) 2022 Imagination Technologies Ltd. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import importlib

#from . import topk

from .topk import Topk, MultiLabelTopk
from .save_image import SaveImages


def build_postprocess(config):
    config = copy.deepcopy(config)
    model_name = config.pop("name")
    mod = importlib.import_module(__name__)
    postprocess_func = getattr(mod, model_name)(**config)
    return postprocess_func
