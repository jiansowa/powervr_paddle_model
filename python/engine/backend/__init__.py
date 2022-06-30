#from .paddle_engine import PaddleInference
from .simulator import SimClasRuntime
from .pvr_grpc import PowerVR_Infer_Cmdline
from .powervr import PowerVR_Infer_gRPC

def build_inference(config):
    backend_type = config['Model']['backend']
    assert backend_type in ['powervr', 'paddle', 'clas_sim', 'powervr_grpc'
            ],"backend type should be 'clas_sim', 'powervr', 'paddle'"
    if backend_type == 'powervr':
        powervr_config = config['Model']['PowerVR']
        return PowerVR_Infer(powervr_config)
    if backend_type == 'powervr_grpc':
        pvr_grpc_config = config['Model']['PowerVR_gRPC']
        return PowerVR_Infer_gRPC(pvr_grpc_config)
    #elif backend_type == 'paddle':
    #    paddle_config = config['Model']['Paddle']
    #    return PaddleInference(paddle_config)
    elif backend_type == 'clas_sim':
        sim_config = config['Model']['ClasSim']
        return SimClasRuntime(sim_config)
        
