#!/usr/bin/python3
"""Install the Arc demo's quiet boot/session profile, with an exact rollback.

Run as root with the checked-out turret repository and expected commit. The
installer does not reboot, restart GDM, select a model, or issue motor commands.
"""
import argparse
import configparser
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time


def quiet_grub(text):
    if not text.startswith('set timeout=3\nset default=0\n'):
        raise ValueError('Unexpected A/B GRUB configuration; inspect before changing it')
    text = text.replace('set timeout=3\n', 'set timeout_style=hidden\nset timeout=0\n', 1)
    background = '''if background_image ($spring_esp)/EFI/spring/background.png; then
    set color_normal=light-gray/black
    set menu_color_normal=light-gray/black
    set menu_color_highlight=black/light-cyan
fi'''
    if text.count(background) != 1:
        raise ValueError('Unexpected GRUB branding block')
    text = text.replace(background, 'set color_normal=white/black\nclear')
    before = 'quiet splash plymouth.ignore-serial-consoles console=tty0 console=ttyS0,115200n8'
    if text.count(before) != 2:
        raise ValueError('Expected both existing A/B kernel entries')
    return text.replace(before, before + ' loglevel=3 systemd.show_status=false vt.global_cursor_default=0')


def ini_update(text, section, changes):
    config = configparser.ConfigParser(interpolation=None)
    config.optionxform = str
    config.read_string(text)
    if not config.has_section(section):
        config.add_section(section)
    for key, value in changes.items():
        config[section][key] = value
    out = io.StringIO()
    config.write(out)
    return out.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('repository', type=Path)
    parser.add_argument('commit')
    args = parser.parse_args()
    assert os.geteuid() == 0
    assert Path('/sys/class/dmi/id/board_name').read_text().strip() == 'B550M WiFi'
    assert Path('/sys/class/graphics/fb0/name').read_text().strip() == 'xedrmfb'
    repo = args.repository.resolve()
    commit = subprocess.check_output(['git', '-c', 'safe.directory='+str(repo), '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == args.commit
    assert not subprocess.check_output(['git', '-c', 'safe.directory='+str(repo), '-C', str(repo), 'status', '--porcelain'], text=True)
    source = repo/'deploy/startup'
    grub = Path('/boot/efi/EFI/spring/grub.cfg')
    grubenv = Path('/boot/efi/EFI/spring/grubenv')
    original_env = grubenv.read_bytes()
    config_file = Path('/home/spring/.config/turret-demo/config.json')
    original_config = config_file.read_bytes()
    kernel = Path('/boot/vmlinuz').resolve().name.removeprefix('vmlinuz-')
    initrd = Path('/boot/initrd.img').resolve()
    assert initrd.name == 'initrd.img-'+kernel
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    backup = Path('/var/backups')/('spring-clean-boot-'+stamp)
    backup.mkdir(mode=0o700)
    records = {}

    def remember(path):
        path = Path(path)
        if str(path) in records:
            return
        assert not path.is_symlink(), str(path)
        record = {'exists': path.exists()}
        if path.exists():
            st = path.stat()
            record.update(mode=st.st_mode & 0o777, uid=st.st_uid, gid=st.st_gid,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            dest = backup/'files'/str(path).lstrip('/')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        records[str(path)] = record
        (backup/'files.json').write_text(json.dumps(records, indent=2)+'\n')

    def write(path, data, mode=0o644, owner=None):
        path = Path(path)
        remember(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(path.name+'.spring-new')
        staged.write_bytes(data if isinstance(data, bytes) else data.encode())
        staged.chmod(mode)
        if owner:
            os.chown(staged, *owner)
        os.replace(staged, path)

    account = '/org/freedesktop/Accounts/User1000'
    interface = 'org.freedesktop.Accounts.User'
    sessions = {}
    for prop in ('XSession', 'Session', 'SessionType'):
        value = subprocess.check_output(['busctl', 'get-property', 'org.freedesktop.Accounts', account, interface, prop], text=True)
        sessions[prop] = shlex.split(value)[1]
    (backup/'sessions.json').write_text(json.dumps(sessions, indent=2)+'\n')
    enabled = subprocess.run(['systemctl', 'is-enabled', 'spring-native-logo.service'], capture_output=True, text=True).stdout.strip() == 'enabled'
    (backup/'before.json').write_text(json.dumps({'commit': commit, 'kernel': kernel, 'native_logo_enabled': enabled}, indent=2)+'\n')
    changed = False
    try:
        candidate = quiet_grub(grub.read_text())
        check = backup/'candidate-grub.cfg'
        check.write_text(candidate)
        subprocess.run(['grub-script-check', str(check)], check=True)
        remember(initrd)
        changed = True
        write(grub, candidate)
        for src, dest, mode in (
            ('clean-boot/spring-silicon.script', '/usr/share/plymouth/themes/spring-silicon/spring-silicon.script', 0o644),
            ('clean-boot/show-native-logo.py', '/usr/local/lib/spring-turret/show-native-logo.py', 0o644),
            ('clean-boot/spring-native-logo.service', '/etc/systemd/system/spring-native-logo.service', 0o644),
            ('clean-boot/openbox.xml', '/etc/spring-turret-kiosk/openbox.xml', 0o644),
            ('kiosk-session.sh', '/usr/local/bin/spring-turret-session', 0o755),
            ('spring-turret-session.desktop', '/usr/share/xsessions/spring-turret.desktop', 0o644),
        ):
            write(dest, (source/src).read_bytes(), mode)
        gdm = Path('/etc/gdm3/custom.conf')
        write(gdm, ini_update(gdm.read_text(), 'daemon', {'AutomaticLoginEnable':'true', 'AutomaticLogin':'spring', 'DefaultSession':'spring-turret.desktop'}))
        dmrc = Path('/home/spring/.dmrc')
        write(dmrc, ini_update(dmrc.read_text() if dmrc.exists() else '', 'Desktop', {'Session':'spring-turret'}), 0o644, (1000,1000))
        remember('/var/lib/AccountsService/users/spring')
        for prop, value in {'XSession':'spring-turret', 'Session':'spring-turret', 'SessionType':'x11'}.items():
            subprocess.run(['busctl', 'call', 'org.freedesktop.Accounts', account, interface, 'Set'+prop, 's', value], check=True)
        subprocess.run(['systemctl', 'daemon-reload'], check=True)
        subprocess.run(['systemd-analyze', 'verify', '/etc/systemd/system/spring-native-logo.service'], check=True)
        subprocess.run(['systemctl', 'enable', 'spring-native-logo.service'], check=True)
        subprocess.run(['desktop-file-validate', '/usr/share/xsessions/spring-turret.desktop'], check=True)
        print(json.dumps({'phase':'rebuilding-initramfs','kernel':kernel,'backup':str(backup)}), flush=True)
        subprocess.run(['update-initramfs', '-u', '-k', kernel], check=True)
        archive = subprocess.check_output(['lsinitramfs', str(initrd)], text=True)
        assert 'usr/share/plymouth/themes/spring-silicon/spring-silicon.script' in archive
        assert grubenv.read_bytes() == original_env, 'A/B health state changed'
        assert config_file.read_bytes() == original_config, 'Inference/motor config changed'
        receipt = {'commit':commit, 'backup':str(backup), 'kernel':kernel,
                   'installed':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in records},
                   'ab_health_preserved':True, 'inference_motor_config_preserved':True,
                   'next_session':'spring-turret (X11)', 'reboot_performed':False,
                   'firmware_logo':'BIOS change still required; no supported Linux firmware setting exposed'}
        (backup/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt), flush=True)
    except Exception:
        if changed:
            if not enabled:
                subprocess.run(['systemctl','disable','spring-native-logo.service'],check=False)
            for name, record in reversed(list(records.items())):
                path = Path(name)
                if record['exists']:
                    shutil.copy2(backup/'files'/name.lstrip('/'), path)
                    os.chown(path, record['uid'], record['gid'])
                    path.chmod(record['mode'])
                else:
                    path.unlink(missing_ok=True)
            for prop, value in sessions.items():
                subprocess.run(['busctl','call','org.freedesktop.Accounts',account,interface,'Set'+prop,'s',value], check=False)
            subprocess.run(['systemctl','daemon-reload'],check=False)
            print('Installation failed; previous boot files, initramfs and session restored.', flush=True)
        raise


if __name__ == '__main__':
    main()
