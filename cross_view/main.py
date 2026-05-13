from modules.config_loader.yaml_loader import load_config
from pipelines.cross_view_tracking import CrossViewTracking

if __name__ == '__main__':
    config = load_config('../cfg/awl_cfg.yaml')
    input_config = load_config('../cfg/awl_input_cfg.yaml')
    entrance_tracking = CrossViewTracking(config, input_config)
    entrance_tracking.run(output_dir='awl_video', file_name='results_reid.mp4')
    # entrance_tracking.run(output_dir='awl_video', file_name='results.mp4')