#!/usr/bin/env python3
"""Allowlisted private recovery export. Never copies login/browser credentials.

This is a same-architecture runtime backup, not an OS flasher or motor controller.
Run as root on the named device; archives must NOT be uploaded to the public repo.
"""
import argparse
import hashlib
import fnmatch
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tarfile
import time

USER = Path('/home/spring')
DEMO = USER/'.local/share/turret-demo'
FRONTEND = USER/'.local/share/spring-turret-frontend'
HOSTS = {'arc':'spring-edge-turret', 'thor':'agxthor-5'}
EXCLUDES = ('__pycache__', '*.pyc', '*.pyo')
ARC_MODELS = ('checkpoint', 'sam31_w4a4_bundle', 'sam31_tracking_bundle',
              'sam31_tracking_native_bundle', 'sam31_tracking_v18_bundle',
              'sam31_mask_bundle', 'sam31_mask_compiled_bundle')
ARC_BOOT_FILES = (
    '/home/spring/.dmrc', '/var/lib/AccountsService/users/spring',
    '/usr/local/bin/spring-turret-session',
    '/usr/share/xsessions/spring-turret.desktop',
    '/etc/spring-turret-kiosk/openbox.xml',
    '/usr/local/lib/spring-turret/show-native-logo.py',
    '/etc/systemd/system/spring-native-logo.service',
    '/usr/share/plymouth/themes/spring-silicon/spring-silicon.script',
    '/usr/share/plymouth/themes/spring-silicon/spring-silicon.plymouth',
    '/usr/share/plymouth/themes/spring-silicon/spring-silicon-plymouth.png',
)


def arc_model_paths(inference):
    unknown = {key for key in inference if key.endswith('_bundle')} - set(ARC_MODELS)
    if unknown:
        raise ValueError(f'Unrecognized model bundle settings; add explicit export coverage: {sorted(unknown)}')
    paths = []
    for key in ARC_MODELS:
        value = inference.get(key)
        if value is None:
            if key == 'checkpoint':
                raise ValueError('Missing checkpoint')
            continue
        path = Path(value)
        if not path.is_relative_to(DEMO) or path == DEMO or '..' in path.parts:
            raise ValueError(f'Model points outside the expected runtime root: {path}')
        paths.append(path)
    return paths


def symlink_inventory(paths):
    """Reject implicit dependencies outside the explicit backup roots."""
    links = {}
    roots = [p.resolve(strict=True) for p in paths if not p.is_symlink()]
    for root in paths:
        entries = [root]
        if root.is_dir() and not root.is_symlink():
            for directory, dirs, files in os.walk(root, followlinks=False):
                dirs[:] = [n for n in dirs if not any(fnmatch.fnmatch(n, x) for x in EXCLUDES)]
                entries.extend(Path(directory)/n for n in dirs+files)
        for path in entries:
            if not path.is_symlink():
                continue
            target = path.resolve(strict=True)
            covered = any(target == p or (p.is_dir() and target.is_relative_to(p)) for p in roots)
            if not covered and target != Path('/usr/bin/python3.12'):
                raise ValueError(f'Unbundled symlink dependency: {path} -> {target}')
            links[str(path)] = {'target':str(target), 'provided_by':'archive' if covered else 'base OS'}
    return links


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source,'sha256').hexdigest()


def python_inventory(command):
    code = ('import importlib.metadata as m,json,sys; '
            'print(json.dumps({"python":sys.version,"packages":sorted('
            '[(d.metadata["Name"],d.version) for d in m.distributions()])}))')
    return json.loads(run(*command,'-c',code))


def paths_for(profile,config):
    state_root=DEMO if profile=='arc' else Path('/var/lib/spring-turret-demo')
    for path in (Path(config['servo']['calibration_file']),Path(config['tracking']['geometry_file'])):
        if not path.is_relative_to(state_root) or '..' in path.parts:
            raise ValueError(f'Configuration points outside the expected state root: {path}')
    models = arc_model_paths(config['inference']) if profile == 'arc' else []
    paths = [Path('/etc/udev/rules.d/99-spring-turret-controller.rules'),
        Path('/etc/spring-turret-direct-link.nft'),
        Path('/etc/NetworkManager/dispatcher.d/pre-up.d/90-turret-direct-link')]
    if profile == 'arc':
        paths += [USER/'.config/turret-demo/config.json',
            USER/'.config/autostart/spring-turret-kiosk.desktop',
            Path('/etc/gdm3/custom.conf'),Path('/var/lib/systemd/linger/spring'),
            USER/'snap/firefox/common/spring-turret-kiosk-profile/user.js',
            DEMO/'venv',DEMO/'inference-venv',DEMO/'opencl',
            Path('/opt/spring-provision/offline-apt')]
        for name in ('spring-turret-demo','spring-turret-frontend',
                     'spring-turret-inference-startup','spring-turret-kiosk'):
            unit=USER/f'.config/systemd/user/{name}.service'
            paths.append(unit)
            drop=Path(str(unit)+'.d')
            if drop.exists():paths.append(drop)
        for name in ('runtime','scripts','src'):
            paths.append(FRONTEND/name)
        paths.append((FRONTEND/'runtime').resolve(strict=True))
        headers=USER/'.local/share/uv/python/cpython-3.12-linux-x86_64-gnu'
        paths += [headers,headers.resolve(strict=True)]
        paths += models
        # The session marker makes these a required set on the quiet-boot install.
        # EFI slot selection/grubenv and the target's initramfs are not portable.
        if Path('/usr/share/xsessions/spring-turret.desktop').exists():
            paths += [Path(name) for name in ARC_BOOT_FILES]
        # Frozen bundles link to these exact runtime trees. Do not follow arbitrary
        # future links (which could accidentally include credentials or large worktrees).
        paths += [USER/'sam3_1',
            DEMO/'sleepy-mask-update-20260909.bDg4xS/bundle/attention8']
    else:
        paths += [Path('/etc/spring-turret-demo.json'),Path('/etc/docker/daemon.json'),
            Path('/etc/udev/rules.d/99-spring-turret.rules'),
            Path('/etc/systemd/system/spring-turret-demo.service'),
            Path('/etc/systemd/system/spring-turret-demo.service.d'),
            Path('/etc/systemd/system/spring-turret-inference-startup.service'),
            Path('/opt/spring/turret-demo/run-container.sh'),
            Path('/opt/spring/turret-demo/run-container-overhead.sh'),
            Path('/opt/spring/turret-demo/start-inference.py'),
            Path('/opt/spring/turret-demo/models/sam3.1_multiplex.pt')]
    paths += [Path(config['servo']['calibration_file']),Path(config['tracking']['geometry_file'])]
    paths = sorted(set(paths))
    for path in paths:
        if not path.exists() and not path.is_symlink():
            raise FileNotFoundError(f'Required deployment component is missing: {path}')
        if not path.is_absolute() or '..' in path.parts:
            raise ValueError(f'Unsafe deployment path: {path}')
    return paths


def capture(profile,output,commit):
    if os.geteuid()!=0:raise PermissionError('Capture requires sudo for Docker/system configuration')
    if platform.node()!=HOSTS[profile]:raise ValueError('Wrong host for requested profile')
    if not re.fullmatch(r'[0-9a-f]{40}',commit):raise ValueError('Supply the reviewed full source commit')
    output=Path(output).absolute()
    if output.exists():raise FileExistsError('Use a NEW output directory; exports are immutable')
    config_path=USER/'.config/turret-demo/config.json' if profile=='arc' else Path('/etc/spring-turret-demo.json')
    config=json.loads(config_path.read_text())
    paths=paths_for(profile,config)
    links=symlink_inventory(paths)
    output.mkdir(mode=0o700,parents=True)
    os.umask(0o077)
    inventory={'schema':1,'profile':profile,'hostname':platform.node(),
        'captured_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'architecture':platform.machine(),'kernel':platform.release(),
        'os_release':Path('/etc/os-release').read_text(),'source_commit':commit,
        'boot_arguments':Path('/proc/cmdline').read_text().strip(),
        'packages':run('dpkg-query','-W','-f=${binary:Package}\t${Version}\n').splitlines(),
        'config':config,'zeros':json.loads(Path(config['servo']['calibration_file']).read_text()),
        'roots':[str(p) for p in paths],
        'files':{},'archives':{},'symlinks':links,
        'exclusions':['credentials','SSH/Tailscale identity','browser profile except user.js',
                      'compiler caches','in-memory target/Start state','operating system image']}
    for path in paths:
        if path.is_file() and not path.is_symlink():inventory['files'][str(path)]=digest(path)
    inventory['wired_connection']=run('nmcli','-g',
        'connection.id,connection.interface-name,802-3-ethernet.mac-address,ipv4.addresses,ipv4.never-default,ipv6.method',
        'connection','show','turret-direct-ethernet').splitlines()
    if profile=='arc':
        inventory['backend_python']=python_inventory([str(DEMO/'venv/bin/python')])
        inventory['inference_python']=python_inventory([str(DEMO/'inference-venv/bin/python')])
        inventory['node']=run(str(FRONTEND/'runtime/bin/node'),'--version')
        inventory['firefox_snap']=run('snap','list','firefox')
        inventory['boot_restore'] = {
            'quiet_session': Path('/usr/share/xsessions/spring-turret.desktop').exists(),
            'target_actions': ['Reapply Firefox hold', 'Enable restored system/user units',
                               'Transform target A/B GRUB in place and rebuild target initramfs'],
            'excluded': ['EFI slot configuration and grubenv', 'source initramfs',
                         'snapd state and automatic-update hold database'],
        }
    else:
        image=json.loads(run('docker','inspect','spring-turret-demo'))[0]
        inventory['docker_image_id']=image['Image']
        inventory['docker_image_tag']=image['Config']['Image']
        inventory['inference_python']=python_inventory(['docker','exec','spring-turret-demo','python'])
        inventory['l4t_release']=Path('/etc/nv_tegra_release').read_text()
        inventory['power_mode']=run('nvpmodel','-q')
        # Save the immutable ID, not a tag which another process could retarget.
        print('Saving exact Thor container image',flush=True)
        with (output/'container.tar.zst').open('xb') as target:
            source=subprocess.Popen(['docker','save',image['Image']],stdout=subprocess.PIPE)
            try:
                subprocess.run(['zstd','-T2','-3'],stdin=source.stdout,stdout=target,check=True)
            finally:source.stdout.close()
            if source.wait()!=0:raise RuntimeError('Docker image export failed')
    print('Saving allowlisted runtime, models and host configuration',flush=True)
    subprocess.run(['tar','--xattrs','--acls','--numeric-owner',
        *(f'--exclude={name}' for name in EXCLUDES),'-I','zstd -T2 -3',
        '-cf',str(output/'runtime.tar.zst'),'-C','/',*(str(p).lstrip('/') for p in paths)],check=True)
    for archive in sorted(output.glob('*.tar.zst')):
        inventory['archives'][archive.name]={'sha256':digest(archive),'bytes':archive.stat().st_size}
    for name, expected in inventory['files'].items():
        if digest(name) != expected:
            raise RuntimeError(f'Configuration/model changed during capture: {name}; recapture')
    (output/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    (output/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.name}\n'
        for p in sorted(output.iterdir()) if p.name!='SHA256SUMS'))
    print(json.dumps({'output':str(output),'inventory_sha256':digest(output/'inventory.json'),
                      'archives':inventory['archives']}),flush=True)


def verify(output):
    output=Path(output)
    manifest=json.loads((output/'inventory.json').read_text())
    if manifest.get('schema')!=1:raise ValueError('Unknown bundle schema')
    # SHA256SUMS must itself be obtained from a trusted transfer / public lock.
    checksums={}
    for line in (output/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split('  ',1)
        if name not in ('inventory.json','runtime.tar.zst','container.tar.zst'):
            raise ValueError('Unexpected checksum filename')
        if name in checksums:raise ValueError('Duplicate checksum filename')
        checksums[name]=expected
    if set(checksums)!={'inventory.json',*manifest['archives']} or 'runtime.tar.zst' not in checksums:
        raise ValueError('Incomplete checksum manifest')
    if digest(output/'inventory.json')!=checksums['inventory.json']:
        raise ValueError('Inventory checksum mismatch')
    for name,info in manifest['archives'].items():
        if name not in ('runtime.tar.zst','container.tar.zst'):raise ValueError('Unexpected archive')
        path=output/name
        if (path.stat().st_size!=info['bytes'] or info['sha256']!=checksums[name]
                or digest(path)!=info['sha256']):
            raise ValueError(f'Archive digest mismatch: {name}')
        subprocess.run(['zstd','--test',str(path)],check=True)
    required={p.lstrip('/') for p in manifest['roots']}
    hashed={p.lstrip('/'):value for p,value in manifest['files'].items()}
    decoder=subprocess.Popen(['zstd','-dc',str(output/'runtime.tar.zst')],stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=decoder.stdout,mode='r|') as archive:
            for member in archive:
                name=member.name.rstrip('/')
                if name.startswith('/') or '..' in Path(name).parts:
                    raise ValueError(f'Unsafe archive member: {name}')
                required.discard(name)
                if name in hashed:
                    content=archive.extractfile(member)
                    if content is None or hashlib.file_digest(content,'sha256').hexdigest()!=hashed.pop(name):
                        raise ValueError(f'Archived file mismatch: {name}')
    finally:
        decoder.stdout.close()
        decoder_status=decoder.wait()
    if decoder_status!=0:raise RuntimeError('Archive decoder failed')
    if required or hashed:raise ValueError(f'Missing archive roots/files: {sorted(required | hashed.keys())}')
    print('Archive hashes, streams, required roots and configuration/model contents verified',flush=True)


def public_lock(output):
    """Publish provenance and hashes, not the private runtime or machine identity."""
    output=Path(output)
    inventory_path=output if output.is_file() else output/'inventory.json'
    inventory=json.loads(inventory_path.read_text())
    names=('schema','profile','hostname','architecture','kernel','os_release',
           'source_commit','captured_utc','config','zeros','roots','files','archives',
           'wired_connection','backend_python','inference_python','node',
           'firefox_snap','boot_restore','docker_image_id','docker_image_tag','l4t_release','power_mode')
    result={name:inventory[name] for name in names if name in inventory}
    result['inventory_sha256']=digest(inventory_path)
    result['host_packages']=inventory['packages']
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='operation',required=True)
    export=sub.add_parser('capture');export.add_argument('profile',choices=HOSTS)
    export.add_argument('--output',required=True);export.add_argument('--source-commit',required=True)
    check=sub.add_parser('verify');check.add_argument('directory')
    lock=sub.add_parser('public-lock');lock.add_argument('directory')
    args=parser.parse_args()
    if args.operation=='capture':capture(args.profile,args.output,args.source_commit)
    elif args.operation=='verify':verify(args.directory)
    else:public_lock(args.directory)


if __name__=='__main__':main()
