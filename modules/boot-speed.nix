# 33.8s -> 20.6s:
# From analysing `systemd-analyze critical-chain`
{ lib, ... }:
{
  # unlock more parallelism
  boot.initrd.kernelModules = [ "nvgpu" ];
  boot.initrd.extraFirmwarePaths = [ "nvidia/ga10b" ];

  boot.loader.timeout = 0;

  # don't make network block anything
  networking.dhcpcd.wait = "background";
  systemd.services."wpa_supplicant-wlP1p1s0".before = lib.mkForce [ ];
}
