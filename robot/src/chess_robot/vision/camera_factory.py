"""One place that decides WHICH camera drives the pipeline, from config:
    vision.camera.type: zed | realsense   (default realsense)
Used by VisionSystem and the calibration/capture tools."""
import os


def create_camera(cam_cfg=None, logger=None):
    cfg = cam_cfg or {}
    ctype = str(cfg.get('type', 'realsense')).lower()
    kwargs = dict(width=int(cfg.get('width', 1280)),
                  height=int(cfg.get('height', 720)),
                  fps=int(cfg.get('fps', 30)),
                  logger=logger)
    if cfg.get('serial'):
        kwargs['serial'] = cfg['serial']
    if ctype == 'zed':
        from .camera_zed import ZEDCamera
        return ZEDCamera(**kwargs)
    from .camera import RealSenseDirect
    return RealSenseDirect(**kwargs)


def load_camera_cfg_from_board_config():
    """For standalone tools: read vision.camera from the standard config path."""
    import yaml
    for p in (os.path.expanduser('~/chess_remote/config/board_config.yaml'),
              os.path.join(os.path.dirname(__file__), '..', '..', '..', '..',
                           'config', 'board_config.yaml')):
        p = os.path.abspath(p)
        if os.path.exists(p):
            with open(p) as f:
                cfg = yaml.safe_load(f) or {}
            return (cfg.get('vision') or {}).get('camera') or {}
    return {}
