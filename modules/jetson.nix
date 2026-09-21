{ pkgs, ... }:
{
  nixpkgs.hostPlatform = "aarch64-linux";

  hardware.nvidia-jetpack = {
    enable = true;
    som = "orin-nano";
    super = true; # Orin Nano Super devkit (67 TOPS)
    carrierBoard = "devkit";
    majorVersion = "7";
    maxClock = true;
    firmware.autoUpdate = true;
  };

  services.nvpmodel.profileNumber = 2; # MAXN_SUPER: uncapped
  hardware.graphics.enable = true;
  hardware.nvidia-jetpack.modesetting.enable = false;

  # board's M.2 card is a Realtek RTL8822CE
  # hardware.enableRedistributableFirmware pulls 1.8 GiB of fw
  # These are < 1MB
  hardware.firmware = [
    (pkgs.runCommand "rtw8822ce-firmware" { } ''
      install -Dt $out/lib/firmware/rtw88 \
        ${pkgs.linux-firmware}/lib/firmware/rtw88/rtw8822c_fw.bin \
        ${pkgs.linux-firmware}/lib/firmware/rtw88/rtw8822c_wow_fw.bin
    '')
  ];

  networking.wireless.interfaces = [ "wlP1p1s0" ];
}
