{ config, lib, ... }:
{
  options.appliance.diskDevice = lib.mkOption {
    type = lib.types.str;
    default = "/dev/disk/by-id/nvme-WD_Blue_SN5100_500GB_2605AE400287";
    description = "Installed disk. Find yours with `ls /dev/disk/by-id`.";
  };

  config = {
    boot.loader.systemd-boot.enable = true;

    disko.devices.disk.main = {
      device = config.appliance.diskDevice;
      type = "disk";
      content = {
        type = "gpt";
        partitions = {
          ESP = {
            type = "EF00";
            size = "500M";
            content = {
              type = "filesystem";
              format = "vfat";
              mountpoint = "/boot";
              mountOptions = [ "umask=0077" ];
            };
          };
          swap = {
            size = "16G";
            content = {
              type = "swap";
            };
          };
          root = {
            size = "100%";
            content = {
              type = "filesystem";
              format = "ext4";
              mountpoint = "/";
            };
          };
        };
      };
    };
  };
}
