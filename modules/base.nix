{
  networking.hostName = "nano";
  networking.firewall.enable = false;
  networking.wireless = {
    enable = true;
    networks."jongenet".psk = "dashieristtodessicher";
  };

  services.openssh = {
    enable = true;
    settings.PermitRootLogin = "yes";
  };

  # Deliberately insecure demo login: root / "demo", root SSH allowed
  users.users.root = {
    initialPassword = "demo";
    openssh.authorizedKeys.keys = [
      "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIH6Z4dj1RU+44lXXW1Dw6TW4cLtV/4+qJRO7vFOmyC6C tfc@ai"
    ];
  };
  users.groups.debug = { };

  security.polkit.enable = true;
  security.run0 = {
    enableSudoAlias = true;
    wheelNeedsPassword = false;
  };
  security.sudo.enable = false;

  nix.settings = {
    experimental-features = [
      "nix-command"
      "flakes"
    ];

    trusted-users = [ "@wheel" ];
  };

  system.stateVersion = "26.05";
}
