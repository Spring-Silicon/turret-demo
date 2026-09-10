"""Firewall rename/missing-adapter recovery; never executes nft or network writes."""
from pathlib import Path
import runpy
import unittest

rules = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'deploy/startup/wired-firewall.py'))['rules']


class WiredRecoveryTests(unittest.TestCase):
    def test_renumbered_interface_keeps_same_peer_guard(self):
        config = {'mac':'9c:6b:00:f1:92:49', 'peer':'192.168.249.2'}
        for name in ('enp15s0', 'enp7s0'):
            text = rules(config, {name:config['mac'], 'wlp8s0':'08:f9:7e:aa:0d:c1'}, True)
            self.assertEqual(text.count(f'ip daddr 192.168.249.2 oifname != "{name}" counter reject'), 2)
            self.assertTrue(text.startswith('delete table inet turret_direct_link\n'))
            self.assertNotIn('flush ruleset', text)

    def test_absent_adapter_blocks_peer_on_all_paths(self):
        text = rules({'mac':'9c:6b:00:f1:92:49', 'peer':'192.168.249.2'}, {}, False)
        self.assertEqual(text.count('ip daddr 192.168.249.2 counter reject'), 2)
        self.assertNotIn('delete', text)
        self.assertNotIn('oifname', text)

    def test_invalid_configuration_is_not_executable_input(self):
        with self.assertRaises(ValueError): rules({'mac':'bad', 'peer':'192.168.249.2'}, {}, False)
        with self.assertRaises(ValueError): rules({'mac':'9c:6b:00:f1:92:49', 'peer':'bad'}, {}, False)
        with self.assertRaises(ValueError): rules({'mac':'9c:6b:00:f1:92:49', 'peer':'192.168.249.2'}, {';bad':'9c:6b:00:f1:92:49'}, False)


if __name__ == '__main__': unittest.main()
