#!/usr/bin/env python3
"""Refresh only the direct-peer firewall table using a stable Ethernet MAC."""
import ipaddress
import json
from pathlib import Path
import re
import subprocess
import sys


def rules(config, interfaces, exists):
    mac = config['mac'].lower()
    if not re.fullmatch(r'(?:[0-9a-f]{2}:){5}[0-9a-f]{2}', mac):
        raise ValueError('Invalid Ethernet MAC')
    peer = str(ipaddress.IPv4Address(config['peer']))
    names = [name for name, address in interfaces.items() if address.lower() == mac]
    if len(names) > 1: raise ValueError('Ambiguous Ethernet MAC')
    name = names[0] if names else None
    if name and not re.fullmatch(r'[a-zA-Z0-9_.:-]{1,15}', name):
        raise ValueError('Invalid network interface')
    # No adapter means reject this peer over EVERY route, including Wi-Fi.
    interface = f' oifname != "{name}"' if name else ''
    text = 'delete table inet turret_direct_link\n' if exists else ''
    text += 'table inet turret_direct_link {\n'
    for chain in ('output', 'forward'):
        text += (f' chain {chain} {{ type filter hook {chain} priority 0; policy accept;\n'
                 f'  ip daddr {peer}{interface} counter reject\n }}\n')
    return text + '}\n'


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    interfaces = {p.name: (p/'address').read_text().strip()
                  for p in Path('/sys/class/net').iterdir() if (p/'address').exists()}
    exists = subprocess.run(['/usr/sbin/nft', 'list', 'table', 'inet', 'turret_direct_link'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    # One nft transaction replaces the scoped table without an unprotected gap.
    subprocess.run(['/usr/sbin/nft', '-f', '-'], input=rules(config, interfaces, exists), text=True, check=True)


if __name__ == '__main__': main()
