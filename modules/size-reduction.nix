{
  lib,
  modulesPath,
  pkgs,
  ...
}:

{
  imports = [
    (modulesPath + "/profiles/minimal.nix")

    # requires CONFIG_EROFS_FS, not set in nvidia kernel config
    # more security relevant than for size compared to nvidia pkg sizes
    #(modulesPath + "/profiles/perlless.nix")
  ];

  system.disableInstallerTools = true;
  programs.nano.enable = false;
  programs.fuse.enable = false;
  security.sudo.enable = false;

  services.speechd.enable = false;
  services.pipewire.enable = false;

  fonts.enableDefaultPackages = false;
  fonts.packages = [ pkgs.dejavu_fonts ];

  nix.settings.keep-derivations = false;

  boot.loader.systemd-boot.configurationLimit = lib.mkDefault 2;
  boot.tmp.cleanOnBoot = true;
  services.journald.extraConfig = ''
    SystemMaxUse=250M
    SystemMaxFileSize=50M
  '';
}
