import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/rm1/rm2026/transistor_rm2026_algorithm_visual_ws/install/shm_pytorch_processor_pkg'
