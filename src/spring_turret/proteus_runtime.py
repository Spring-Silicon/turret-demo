"""Private, relocated Proteus Torch/SYCL runtime. No changes to system drivers."""
import os
from pathlib import Path


def prepare(config):
    from .hardware import WorkerSpec
    root = Path(config['proteus_bundle'])
    runtime = root / 'proteus-composed-b580-20260907'
    private = runtime / 'private-host/usr'
    venv = runtime / 'root/env/venv'
    compiler = runtime / 'root/native-matched-2025.3.2/extracted/opt/intel/oneapi/compiler/2025.3'
    lab = root / 'sam31-speed-memoff-20260910'
    cache = Path(config['cache_dir']) / config['model']
    for kind in ('inductor', 'triton', 'extensions'):
        (cache / kind).mkdir(parents=True, exist_ok=True)
    loader = private / 'lib/x86_64-linux-gnu/ld-linux-x86-64.so.2'
    python = private / 'bin/python3.12'
    if not loader.is_file() or not python.is_file():
        raise ValueError('Proteus runtime is incomplete; restore the pinned bundle')
    libraries = ':'.join(map(str, (venv / 'lib', private / 'lib/x86_64-linux-gnu', compiler / 'lib')))
    opencl = str(private / 'lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so')
    env = {**os.environ, 'PYTHONHOME': str(private),
           'PYTHONPATH': ':'.join(map(str, (root / 'sam31-b580-lab/env/site-packages',
               Path(__file__).parent.parent, venv / 'lib/python3.12/site-packages'))),
           'LD_LIBRARY_PATH': libraries, 'CXX': str(compiler / 'bin/icpx'),
           'OCL_ICD_VENDORS': opencl, 'OCL_ICD_FILENAMES': opencl,
           'SYCL_HOME': str(compiler), 'CPATH': str(private / 'include/python3.12'),
           'CPLUS_INCLUDE_PATH': str(private / 'include/python3.12'),
           'PATH': f'{compiler}/bin:{venv}/bin:/usr/bin:/bin',
           'ONEAPI_DEVICE_SELECTOR': f'level_zero:{config.get("device", 0)}',
           'CUDA_VISIBLE_DEVICES': '', 'NVIDIA_VISIBLE_DEVICES': 'void',
           'TRITON_DEFAULT_BACKEND': 'intel', 'USE_PERFLIB': '0',
           'TORCHINDUCTOR_CACHE_DIR': str(cache / 'inductor'),
           'TRITON_CACHE_DIR': str(cache / 'triton'),
           'TORCH_EXTENSIONS_DIR': str(cache / 'extensions'),
           'EFFICIENTSAM3_ONEDNN_LIBRARY_DIR': str(lab / 'cache/native/onednn-3.11.2-build/src'),
           'MAX_JOBS': '2', 'TORCHINDUCTOR_COMPILE_THREADS': '1', 'OMP_NUM_THREADS': '4'}
    command = [str(loader), '--library-path', libraries, str(python), '-u',
               str(Path(__file__).with_name('proteus_worker.py')), '--bundle', str(root),
               '--checkpoint', config['checkpoint'], '--profile', config['model'],
               '--confidence', str(config.get('confidence', .5))]
    if config['model'] == 'efficient-memory':
        command.extend(('--memory-bundle', config['proteus_memory_bundle']))
    return WorkerSpec(command, env)
